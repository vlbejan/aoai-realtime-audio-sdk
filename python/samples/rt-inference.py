import asyncio
import base64
import json
import os
import platform
import random
import sys
import time
import yaml
import numpy as np
import soundfile as sf
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from pathlib import Path
from scipy.signal import resample
from rtclient import (
    InputAudioTranscription,
    RTAudioContent,
    RTClient,
    RTFunctionCallItem,
    RTInputAudioItem,
    RTMessageItem,
    RTResponse,
    ServerVAD,
)
from rtclient.models import NoTurnDetection
from datetime import datetime

random.seed(42)

class DatasetHelper:
    @staticmethod
    def get_dataset_and_file_windows(file_path):
        parts = file_path.split('\\')
        dataset_name = parts[-2]
        audio_file_prefix = parts[-1].split('.')[0]
        return dataset_name, audio_file_prefix

    @staticmethod
    def get_dataset_and_file_linux(file_path):
        dataset_name = os.path.basename(os.path.dirname(file_path))
        audio_file_prefix = os.path.splitext(os.path.basename(file_path))
        return dataset_name, audio_file_prefix

    @staticmethod
    def get_dataset_and_file(file_path):
        if platform.system().lower() == 'linux':
            return DatasetHelper.get_dataset_and_file_linux(file_path)
        else:
            return DatasetHelper.get_dataset_and_file_windows(file_path)

class AudioProcessor:
    TEXT_QUESTION = 'TEXT_QUESTION'
    SYSMSG_TEMPLATE = (
        "You are a helpful assistant that concisely answers questions. You never joke unless asked to. "
        + f"Listen to the audio and answer the following question: '{TEXT_QUESTION}'"
    )

    def __init__(self, file_path, out_dir, provider, n_iter=10, timeout=30, pause=5, api_version=None, deployment=None, model=None):
        self.file_path = file_path
        self.out_dir = out_dir
        self.provider = provider
        self.n_iter = n_iter
        self.timeout = timeout
        self.pause = pause
        self.os_name = platform.system().lower()
        self.iteration = None
        self.output_jsonl_file = None
        self.prompt_question = None
        self.audio_file_prefix = None
        self.ground_truth = None
        self.turn_detection = None
        self.api_version = api_version
        self.deployment = deployment
        self.model = model

    def inject_question_to_system_message(self, question):
        complete_message = AudioProcessor.SYSMSG_TEMPLATE.replace('TEXT_QUESTION', question)
        return complete_message

    def get_dataset_and_file_windows(self, file_path):
        # Split the path into parts using "\\" as the separator
        split_symbol = "\\" if "\\" in file_path else "/"
        parts = file_path.split(split_symbol)
        audio_file_prefix = parts[-1].split('.')[0]
        return audio_file_prefix

    def get_dataset_and_file_linux(self, file_path):
        dataset_name = os.path.basename(os.path.dirname(file_path))
        audio_file_prefix = os.path.splitext(os.path.basename(file_path))
        return dataset_name, audio_file_prefix

    def get_dataset_and_file(self, file_path):
        if self.os_name == 'linux':
            return self.get_dataset_and_file_linux(file_path)
        else:
            return self.get_dataset_and_file_windows(file_path)

    def resample_audio(self, audio_data, original_sample_rate, target_sample_rate):
        number_of_samples = round(len(audio_data) * float(target_sample_rate) / original_sample_rate)
        resampled_audio = resample(audio_data, number_of_samples)
        return resampled_audio.astype(np.int16)

    async def send_audio(self, client: RTClient):
        sample_rate = 24000
        duration_ms = 100
        samples_per_chunk = sample_rate * (duration_ms / 1000)
        bytes_per_sample = 2
        bytes_per_chunk = int(samples_per_chunk * bytes_per_sample)
        extra_params = (
            {
                "samplerate": sample_rate,
                "channels": 1,
                "subtype": "PCM_16",
            }
            if self.file_path.endswith(".raw")
            else {}
        )
        audio_data, original_sample_rate = sf.read(self.file_path, dtype="int16", **extra_params)
        if original_sample_rate != sample_rate:
            audio_data = self.resample_audio(audio_data, original_sample_rate, sample_rate)
        audio_bytes = audio_data.tobytes()
        for i in range(0, len(audio_bytes), bytes_per_chunk):
            chunk = audio_bytes[i : i + bytes_per_chunk]
            await client.send_audio(chunk)
        if isinstance(self.turn_detection, NoTurnDetection):
            await client.commit_audio()
            await client.generate_response()

    async def receive_message_item(self, item: RTMessageItem, output_file_name: str):
        prefix = f"[response={item.response_id}][item={item.id}]"
        async for contentPart in item:
            if contentPart.type == "audio":
                async def collect_audio(audioContentPart: RTAudioContent):
                    audio_data = bytearray()
                    async for chunk in audioContentPart.audio_chunks():
                        audio_data.extend(chunk)
                    return audio_data
                async def collect_transcript(audioContentPart: RTAudioContent):
                    audio_transcript = ""
                    async for chunk in audioContentPart.transcript_chunks():
                        audio_transcript += chunk
                    return audio_transcript
                audio_task = asyncio.create_task(collect_audio(contentPart))
                transcript_task = asyncio.create_task(collect_transcript(contentPart))
                audio_data, audio_transcript = await asyncio.gather(audio_task, transcript_task)
                print(prefix, f"Audio received with length: {len(audio_data)}")
                print(prefix, f"Audio Transcript: {audio_transcript}")
                with open(os.path.join(self.out_dir, f"{item.id}_{contentPart.content_index}.wav"), "wb") as out:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sf.write(out, audio_array, samplerate=24000)
                with open(
                    os.path.join(self.out_dir, f"{item.id}_{contentPart.content_index}.audio_transcript.txt"),
                    "w",
                    encoding="utf-8",
                ) as out:
                    out.write(audio_transcript)
            elif contentPart.type == "text":
                text_data = ""
                async for chunk in contentPart.text_chunks():
                    text_data += chunk
                print(prefix, f"Text: {text_data}")
                with open(
                    os.path.join(self.out_dir, f"{item.id}_{contentPart.content_index}.text.txt"), "w", encoding="utf-8"
                ) as out:
                    out.write(text_data)

    async def receive_function_call_item(self, item: RTFunctionCallItem, output_file_name: str):
        prefix = f"[function_call_item={item.id}]"
        await item
        print(prefix, f"Function call arguments: {item.arguments}")
        with open(os.path.join(self.out_dir, f"{item.id}.function_call.json"), "w", encoding="utf-8") as out:
            out.write(item.arguments)

    async def receive_response(self, client: RTClient, response: RTResponse, output_file_name: str):
        prefix = f"[response={response.id}]"
        async for item in response:
            print(prefix, f"Received item {item.id}")
            if item.type == "message":
                asyncio.create_task(self.receive_message_item(item, output_file_name))
            elif item.type == "function_call":
                asyncio.create_task(self.receive_function_call_item(item, output_file_name))
        print(prefix, "Response completed")
        await client.close()

    async def receive_input_item(self, item: RTInputAudioItem):
        prefix = f"[input_item={item.id}]"
        await item
        print(prefix, f"Transcript: {item.transcript}")
        print(prefix, f"Audio Start [ms]: {item.audio_start_ms}")
        print(prefix, f"Audio End [ms]: {item.audio_end_ms}")

    async def receive_events(self, client: RTClient, output_file_name: str):
        async for event in client.events():
            if event.type == "input_audio":
                asyncio.create_task(self.receive_input_item(event))
            elif event.type == "response":
                asyncio.create_task(self.receive_response(client, event, output_file_name))

    async def receive_messages(self, client: RTClient, output_file_name: str):
        await asyncio.gather(
            self.receive_events(client, output_file_name),
        )

    async def run(self, client: RTClient):
        out_file_prefix = self.get_dataset_and_file(self.file_path)
        out_file_name = out_file_prefix
        sys_message = self.inject_question_to_system_message(self.prompt_question)
        print("Configuring Session...", end="", flush=True)
        await client.configure(
            instructions=sys_message,
            turn_detection=self.turn_detection,
            input_audio_transcription=InputAudioTranscription(model="whisper-1")
        )
        print("Done")
        await asyncio.gather(
            self.send_audio(client),
            self.receive_messages(client, out_file_name)
        )

    def get_env_var(self, var_name: str) -> str:
        value = os.environ.get(var_name)
        if not value:
            raise OSError(f"Environment variable '{var_name}' is not set or is empty.")
        return value

    async def with_azure_openai(self):
        endpoint = self.get_env_var("AZURE_OPENAI_ENDPOINT")
        key = self.get_env_var("AZURE_OPENAI_API_KEY")
        deployment = self.get_env_var("AZURE_OPENAI_DEPLOYMENT")
        async with RTClient(url=endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment) as client:
            try:
                await asyncio.wait_for(
                    self.run(client),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                await client.close()
                print(f"No activity for {self.timeout} seconds. Connection closed.")
            except Exception as e:
                await client.close()
                print(f"An error occurred: {e}")

    async def with_openai(self):
        key = self.get_env_var("OPENAI_API_KEY")
        model = self.get_env_var("OPENAI_MODEL")
        async with RTClient(key_credential=AzureKeyCredential(key), model=model) as client:
            try:
                await asyncio.wait_for(
                    self.run(client),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                await client.close()
                print(f"No activity for {self.timeout} seconds. Connection closed.")
            except Exception as e:
                await client.close()
                print(f"An error occurred: {e}")

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def process_single_file(processor):
    if processor.provider == "azure":
        start_time = time.time()
        for iteration in range(1, processor.n_iter + 1):
            print(f'---\nIteration: {iteration}\n---', end="\r")
            processor.iteration = iteration
            asyncio.run(processor.with_azure_openai())
            time.sleep(processor.pause)
        run_time = time.time() - start_time
        print(f'---\nRuntime for {processor.n_iter} iteration completed in {np.round(run_time / 60, 0)} minutes')
    else:
        start_time = time.time()
        for iteration in range(1, processor.n_iter + 1):
            print(f'---\nIteration: {iteration}\n---', end="\r")
            processor.iteration = iteration
            asyncio.run(processor.with_openai())
            time.sleep(processor.pause)
        run_time = time.time() - start_time
        print(f'---\nRuntime for {processor.n_iter} iteration completed in {np.round(run_time / 60, 0)} minutes')

if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv} <config.yaml>")
        sys.exit(1)
    config_file = sys.argv[1]
    config = load_config(config_file)
    provider = config.get("provider", "azure")
    test_type = config.get("test_type", "quality")
    turn_detection = config.get("turn_detection_type", "ServerVAD")
    endpoint_region = config.get("endpoint_region")
    api_version = config.get("api_version")
    model = config.get("model", None)
    dataset_name = config.get("dataset_name")
    deployment_name = config.get("deployment_name", None)
    n_questions = config.get("n_questions", 200)
    n_iterations = config.get("n_iterations", 1)
    turn_detection_type = ServerVAD() if turn_detection == "ServerVAD" else NoTurnDetection()
    folder_path = fr"C:/Users/vlbejan/Downloads/audio-benchmark-datasets/{dataset_name}"

    if provider == "openai":
        root_out_dir = f"output/benchmark/{test_type}/{provider}/{datetime.now().strftime('%Y%m%d')}"
        del endpoint_region
    else:
        root_out_dir = f"output/benchmark/{test_type}/{provider}/{endpoint_region}/{datetime.now().strftime('%Y%m%d')}"

    json_file_path = os.path.join(folder_path, 'text.json')
    if platform.system().lower() == 'windows':
        json_file_path = os.path.normpath(json_file_path)
    if not os.path.isfile(json_file_path):
        print(f"File {json_file_path} does not exist")
        sys.exit(1)
    if not os.path.isdir(root_out_dir):
        print(f"Root output directory {root_out_dir} does not exist. Creating one...")
        os.makedirs(root_out_dir, exist_ok=True)
    if provider not in ["azure", "openai"]:
        print(f"Provider {provider} needs to be one of 'azure' or 'openai'")
        sys.exit(1)
    if test_type not in ["quality", "parity"]:
        print(f"Test type {test_type} needs to be one of 'quality' or 'parity'")
        sys.exit(1)
    if test_type == "quality" and n_iterations != 1:
        print(f"The number of iterations for {test_type} test should be 1. You set this to {n_iterations} iterations.")
        sys.exit(1)
    if test_type == "parity" and n_iterations <= 1:
        print(f"The number of iterations for {test_type} test should be more than 1. You set this to {n_iterations} iterations.")
        sys.exit(1)

    audio_output_dir = os.path.normpath(os.path.join(root_out_dir, dataset_name))
    if not os.path.isdir(audio_output_dir):
        print(f"Creating audio output directory {audio_output_dir}...")
        os.makedirs(audio_output_dir, exist_ok=True)

    if dataset_name in ("alpaca_audio_test", "openhermes_audio_test") and isinstance(turn_detection_type, NoTurnDetection):
        print(f"Dataset {dataset_name} has audio question, set turn detection to 'ServerVAD'")
        sys.exit(1)
    if dataset_name not in ("alpaca_audio_test", "openhermes_audio_test") and isinstance(turn_detection_type, ServerVAD):
        print(f"Dataset {dataset_name} has audio question, set turn detection to 'NoTurnDetection'")
        sys.exit(1)

    transcript_output_file = os.path.normpath(os.path.join(root_out_dir, f'{provider}_{dataset_name}_rt_output.jsonl'))
    print(f'---\nTrascript file:\n {transcript_output_file}\n---')

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    if provider == "openai":
        sampled_dataset_dir = os.path.abspath(os.path.join(root_out_dir, "..", ".."))
    else:
        sampled_dataset_dir = os.path.abspath(os.path.join(root_out_dir, "..", "..", ".."))
    sampled_dataset_path = Path(os.path.join(sampled_dataset_dir, f'{dataset_name}_selected_entries.jsonl'))

    if sampled_dataset_path.is_file():
        print("File exists")
        sampled_data = []
        with open(sampled_dataset_path, 'r') as f:
            for line in f:
                sampled_data.append(json.loads(line))
    else:
        print(f"File does not exist. Sampling data and saving it to {sampled_dataset_path}\n---")
        if len(data) <= n_questions:
            sampled_data = data.copy()
        else:
            sampled_data = random.sample(data, n_questions)
        with open(sampled_dataset_path, 'w') as f:
            for entry in sampled_data:
                json.dump(entry, f)
                f.write('\n')

    exp_start = time.time()
    for question_id, entry in enumerate(sampled_data):
        print(f'---\n Question number: {question_id}')
        question = entry['prompt']
        ground_truth = entry['answer']
        _, input_file_prefix = DatasetHelper.get_dataset_and_file(entry['audio_path'])
        audio_file_id = entry['id']
        audio_file_name = f'{input_file_prefix}.wav'
        audio_file_path = os.path.join(folder_path, audio_file_name)
        if platform.system().lower() == 'windows':
            transcript_output_file = os.path.normpath(transcript_output_file)
            audio_file_path = os.path.normpath(audio_file_path)
        processor = AudioProcessor(
            file_path=audio_file_path,
            out_dir=audio_output_dir,
            provider=provider,
            n_iter=n_iterations,
            timeout=30,
            pause=10,
            deployment=deployment_name,
            model=model
        )
        processor.prompt_question = question
        processor.ground_truth = ground_truth
        processor.output_jsonl_file = transcript_output_file
        processor.turn_detection = turn_detection_type
        process_single_file(processor)
    exp_run_time = time.time() - exp_start
    print(f'---\nExperiment runtime: {np.round(exp_run_time / 60, 0)} minutes')
