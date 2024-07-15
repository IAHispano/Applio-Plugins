import os
import sys
import random

import gradio as gr
import regex as re

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from rvc.infer.infer import VoiceConverter
from elevenlabs.client import ElevenLabs
from elevenlabs import save

client = ElevenLabs()
i18n = I18nAuto()
voice_converter = VoiceConverter()

model_root = os.path.join(now_dir, "logs")
model_root_relative = os.path.relpath(model_root, now_dir)
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

os.makedirs(custom_embedder_root, exist_ok=True)
custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]


def change_choices():
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def match_index(model_file: str) -> tuple:
    model_files_trip = re.sub(r"\.pth|\.onnx$", "", model_file)
    model_file_name = os.path.split(model_files_trip)[-1]

    if re.match(r".+_e\d+_s\d+$", model_file_name):
        base_model_name = model_file_name.rsplit("_", 2)[0]
    else:
        base_model_name = model_file_name

    sid_directory = os.path.join(model_root_relative, base_model_name)
    directories_to_search = [sid_directory] if os.path.exists(sid_directory) else []
    directories_to_search.append(model_root_relative)
    matching_index_files = []

    for directory in directories_to_search:
        for filename in os.listdir(directory):
            if filename.endswith(".index") and "trained" not in filename:
                name_match = any(
                    name.lower() in filename.lower()
                    for name in [model_file_name, base_model_name]
                )

                folder_match = directory == sid_directory

                if name_match or folder_match:
                    index_path = os.path.join(directory, filename)
                    updated_indexes_list = get_indexes()
                    if index_path in updated_indexes_list:
                        matching_index_files.append(
                            (
                                index_path,
                                os.path.getsize(index_path),
                                " " not in filename,
                            )
                        )
    if matching_index_files:
        matching_index_files.sort(key=lambda x: (-x[2], -x[1]))
        best_match_index_path = matching_index_files[0][0]
        return best_match_index_path

    return ""


custom_embedders = [
    os.path.join(dirpath, filename)
    for dirpath, _, filenames in os.walk(custom_embedder_root_relative)
    for filename in filenames
    if filename.endswith(".pt")
]


def process_input(file_path):
    with open(file_path, "r") as file:
        file_contents = file.read()
    gr.Info(f"The text from the txt file has been loaded!")
    return file_contents, None


def run_tts_script(
    tts_text,
    tts_voice,
    pitch,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    output_tts_path,
    output_rvc_path,
    model_file,
    index_file,
    split_audio,
    autotune,
    clean_audio,
    clean_strength,
    export_format,
    embedder_model,
    embedder_model_custom,
    upscale_audio,
    f0_file,
    api_key,
):
    if os.path.exists(output_tts_path):
        os.remove(output_tts_path)

    if api_key:
        client = ElevenLabs(
            api_key=api_key,
        )
    else:
        client = ElevenLabs()

    tts = client.generate(
        text=tts_text, voice=tts_voice, model="eleven_multilingual_v2"
    )
    save(tts, output_tts_path)

    print(f"TTS with {tts_voice} completed. Output TTS file: '{output_tts_path}'")

    voice_converter.convert_audio(
        f0_up_key=pitch,
        filter_radius=filter_radius,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        hop_length=hop_length,
        f0_method=f0method,
        audio_input_path=output_tts_path,
        audio_output_path=output_rvc_path,
        model_path=model_file,
        index_path=index_file,
        split_audio=split_audio,
        f0_autotune=autotune,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        export_format=export_format,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        upscale_audio=upscale_audio,
        f0_file=f0_file,
    )

    return f"Text {tts_text} synthesized successfully.", output_rvc_path.replace(
        ".wav", f".{export_format.lower()}"
    )


def applio_plugin():
    gr.Markdown(
        """
    ## Elevenlabs TTS Plugin
    This plugin allows you to use the Elevenlabs TTS model to synthesize text into speech. Remeber that elevenlabs is a multilingual TTS model, so you can use it to synthesize text in different languages.
    Languages supported by the model: Chinese, Korean, Dutch, Turkish, Swedish, Indonesian, Filipino, Japanese, Ukrainian, Greek, Czech, Finnish, Romanian, Russian, Danish, Bulgarian, Malay, Slovak, Croatian, Classic Arabic, Tamil, English, Polish, German, Spanish, French, Italian, Hindi and Portuguese.
    """
    )
    default_weight = random.choice(names) if names else ""
    with gr.Row():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(names, key=lambda path: os.path.getsize(path)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            best_default_index_path = match_index(model_file.value)
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=get_indexes(),
                value=best_default_index_path,
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Column():
            refresh_button = gr.Button(i18n("Refresh"))
            unload_button = gr.Button(i18n("Unload Voice"))

            unload_button.click(
                fn=lambda: ({"value": "", "__type__": "update"}),
                inputs=[],
                outputs=[model_file],
            )

            model_file.select(
                fn=match_index,
                inputs=[model_file],
                outputs=[index_file],
            )

    response = client.voices.get_all()
    if hasattr(response, "voices") and isinstance(response.voices, list):
        voices_list = response.voices
        voice_names = [voice.name for voice in voices_list]
    else:
        print("Unexpected response format or missing data.")

    tts_voice = gr.Dropdown(
        label=i18n("TTS Voices"),
        info=i18n("Select the TTS voice to use for the conversion."),
        choices=voice_names,
        interactive=True,
        value=None,
    )

    tts_text = gr.Textbox(
        label=i18n("Text to Synthesize"),
        info=i18n("Enter the text to synthesize."),
        placeholder=i18n("Enter text to synthesize"),
        lines=3,
    )

    api_key = gr.Textbox(
        label=i18n("Optional API Key"),
        placeholder=i18n(
            "Enter your API key (This is necessary if you make a lot of requests)"
        ),
        value="",
        interactive=True,
        info="If you use this feature too much, make sure to grab an API key from ElevenLabs. Simply visit https://elevenlabs.com/ to get yours. Need help? Check out this link: https://elevenlabs.io/docs/api-reference/text-to-speech#authentication",
    )

    txt_file = gr.File(
        label=i18n("Or you can upload a .txt file"),
        type="filepath",
    )

    with gr.Accordion(i18n("Advanced Settings"), open=False):
        with gr.Column():
            output_tts_path = gr.Textbox(
                label=i18n("Output Path for TTS Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
                interactive=True,
            )
            output_rvc_path = gr.Textbox(
                label=i18n("Output Path for RVC Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=i18n("Export Format"),
                info=i18n("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="WAV",
                interactive=True,
            )
            split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split the audio into chunks for inference to obtain better results in some cases."
                ),
                visible=False,
                value=False,
                interactive=True,
            )
            autotune = gr.Checkbox(
                label=i18n("Autotune"),
                info=i18n(
                    "Apply a soft autotune to your inferences, recommended for singing conversions."
                ),
                visible=False,
                value=False,
                interactive=True,
            )
            clean_audio = gr.Checkbox(
                label=i18n("Clean Audio"),
                info=i18n(
                    "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                ),
                visible=False,
                value=True,
                interactive=True,
            )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Clean Strength"),
                info=i18n(
                    "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                ),
                visible=False,
                value=0.5,
                interactive=True,
            )
            upscale_audio = gr.Checkbox(
                label=i18n("Upscale Audio"),
                info=i18n(
                    "Upscale the audio to a higher quality, recommended for low-quality audios. (It could take longer to process the audio)"
                ),
                visible=False,
                value=False,
                interactive=True,
            )
            pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Pitch"),
                info=i18n(
                    "Set the pitch of the audio, the higher the value, the higher the pitch."
                ),
                value=0,
                interactive=True,
            )
            filter_radius = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n("Filter Radius"),
                info=i18n(
                    "If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."
                ),
                value=3,
                step=1,
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search Feature Ratio"),
                info=i18n(
                    "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                ),
                value=0.75,
                interactive=True,
            )
            rms_mix_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Volume Envelope"),
                info=i18n(
                    "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                ),
                value=1,
                interactive=True,
            )
            protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect Voiceless Consonants"),
                info=i18n(
                    "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                ),
                value=0.5,
                interactive=True,
            )
            hop_length = gr.Slider(
                minimum=1,
                maximum=512,
                step=1,
                label=i18n("Hop Length"),
                info=i18n(
                    "Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."
                ),
                value=128,
                interactive=True,
            )
            f0method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n(
                    "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                ),
                choices=[
                    "pm",
                    "harvest",
                    "dio",
                    "crepe",
                    "crepe-tiny",
                    "rmvpe",
                    "fcpe",
                    "hybrid[rmvpe+fcpe]",
                ],
                value="rmvpe",
                interactive=True,
            )
            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for learning speaker embedding."),
                choices=["hubert", "contentvec"],
                value="hubert",
                interactive=True,
            )
            with gr.Column(visible=False) as embedder_custom:
                with gr.Accordion(i18n("Custom Embedder"), open=True):
                    embedder_upload_custom = gr.File(
                        label=i18n("Upload Custom Embedder"),
                        type="filepath",
                        interactive=True,
                    )
                    embedder_custom_refresh = gr.Button(i18n("Refresh"))
                    embedder_model_custom = gr.Dropdown(
                        label=i18n("Custom Embedder"),
                        info=i18n(
                            "Select the custom embedder to use for the conversion."
                        ),
                        choices=sorted(custom_embedders),
                        interactive=True,
                        allow_custom_value=True,
                    )
            f0_file = gr.File(
                label=i18n(
                    "The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls."
                ),
                visible=True,
            )

    convert_button1 = gr.Button(i18n("Convert"))

    with gr.Row():
        vc_output1 = gr.Textbox(label=i18n("Output Information"))
        vc_output2 = gr.Audio(label=i18n("Export Audio"))

    refresh_button.click(
        fn=change_choices,
        inputs=[],
        outputs=[model_file, index_file],
    )
    txt_file.upload(
        fn=process_input,
        inputs=[txt_file],
        outputs=[tts_text, txt_file],
    )
    convert_button1.click(
        fn=run_tts_script,
        inputs=[
            tts_text,
            tts_voice,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0method,
            output_tts_path,
            output_rvc_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            clean_audio,
            clean_strength,
            export_format,
            embedder_model,
            embedder_model_custom,
            upscale_audio,
            f0_file,
            api_key,
        ],
        outputs=[vc_output1, vc_output2],
    )
