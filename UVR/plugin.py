import os

import gradio as gr
from audio_separator.separator import Separator

plugin_folder = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(os.getcwd(), "rvc", "models", "uvr")
audios_folder = os.path.join(os.getcwd(), "assets", "audios", "uvr")


def get_models_by_type(arch_type):
    try:
        separator = Separator(
            info_only=True,
            model_file_dir=models_folder,
        )
        all_models = separator.list_supported_model_files()
        models_for_arch = all_models.get(arch_type, {})
        return [
            info["filename"]
            for info in models_for_arch.values()
            if info.get("filename")
        ]
    except Exception:
        return []


def run_uvr(
    audio_path,
    audio_upload,
    output_format,
    output_bitrate,
    output_dir,
    invert_spect,
    normalization,
    amplification,
    single_stem,
    sample_rate,
    chunk_duration,
    use_autocast,
    use_soundfile,
    # VR
    vr_model,
    vr_batch_size,
    vr_window_size,
    vr_aggression,
    vr_enable_tta,
    vr_high_end_process,
    vr_enable_post_process,
    vr_post_process_threshold,
    # MDX
    mdx_model,
    mdx_segment_size,
    mdx_overlap,
    mdx_batch_size,
    mdx_hop_length,
    mdx_enable_denoise,
    # MDXC
    mdxc_model,
    mdxc_segment_size,
    mdxc_override_model_segment_size,
    mdxc_overlap,
    mdxc_batch_size,
    mdxc_pitch_shift,
    # Demucs
    demucs_model,
    demucs_segment_size,
    demucs_shifts,
    demucs_overlap,
    demucs_segments_enabled,
    tab_selected,
):
    audio = audio_path if audio_path and audio_path.strip() != "" else audio_upload
    if not audio:
        raise gr.Error("No audio provided. Please upload a file or provide a path.")

    if tab_selected == "VR":
        model = vr_model
    elif tab_selected == "MDX":
        model = mdx_model
    elif tab_selected == "MDXC":
        model = mdxc_model
    elif tab_selected == "Demucs":
        model = demucs_model
    else:
        model = vr_model

    if not model:
        raise gr.Error("No model selected. Please choose a model before running.")

    if single_stem == "All Stems (Default)":
        single_stem = None

    os.makedirs(output_dir, exist_ok=True)

    separator = Separator(
        model_file_dir=models_folder,
        output_dir=output_dir,
        output_format=output_format,
        output_bitrate=output_bitrate if output_bitrate.strip() != "" else None,
        normalization_threshold=float(normalization),
        amplification_threshold=float(amplification),
        output_single_stem=single_stem,
        invert_using_spec=invert_spect,
        sample_rate=int(sample_rate),
        chunk_duration=int(chunk_duration) if int(chunk_duration) > 0 else None,
        use_autocast=use_autocast,
        use_soundfile=use_soundfile,
        vr_params={
            "batch_size": int(vr_batch_size),
            "window_size": int(vr_window_size),
            "aggression": int(vr_aggression),
            "enable_tta": vr_enable_tta,
            "enable_post_process": vr_enable_post_process,
            "post_process_threshold": float(vr_post_process_threshold),
            "high_end_process": vr_high_end_process,
        },
        mdx_params={
            "hop_length": int(mdx_hop_length),
            "segment_size": int(mdx_segment_size),
            "overlap": float(mdx_overlap),
            "batch_size": int(mdx_batch_size),
            "enable_denoise": mdx_enable_denoise,
        },
        mdxc_params={
            "segment_size": int(mdxc_segment_size),
            "batch_size": int(mdxc_batch_size),
            "overlap": int(mdxc_overlap),
            "override_model_segment_size": mdxc_override_model_segment_size,
            "pitch_shift": int(mdxc_pitch_shift),
        },
        demucs_params={
            "segment_size": int(demucs_segment_size)
            if str(demucs_segment_size).lower() != "default"
            else "Default",
            "shifts": int(demucs_shifts),
            "overlap": float(demucs_overlap),
            "segments_enabled": demucs_segments_enabled,
        },
    )
    separator.load_model(model_filename=model)
    output_filenames = separator.separate(audio)
    return [os.path.join(output_dir, f) for f in output_filenames]


def applio_plugin():
    audio_upload = gr.Audio(
        label="Upload audio",
        sources=["upload", "microphone"],
        type="filepath",
        interactive=True,
    )
    audio_path = gr.Textbox(
        label="Input audio path",
        placeholder="Paste path here...",
        interactive=True,
    )

    single_stem = gr.Radio(
        label="Single stem",
        info="Extract specific stem. 'All Stems' outputs all stems supported by the selected model.",
        choices=[
            "All Stems (Default)",
            "Instrumental",
            "Vocals",
            "Drums",
            "Bass",
            "Guitar",
            "Piano",
            "Other",
        ],
        value="All Stems (Default)",
        interactive=True,
    )

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            invert_spect = gr.Checkbox(
                label="Invert spectrogram",
                info="Inverts secondary stem using spectrogram-based processing instead of waveform. Slightly slower but may improve quality.",
                value=False,
                interactive=True,
            )
            use_autocast = gr.Checkbox(
                label="Use Autocast (PyTorch)",
                info="Uses mixed precision (FP16) to speed up GPU inference.",
                value=False,
                interactive=True,
            )
            use_soundfile = gr.Checkbox(
                label="Use Soundfile",
                info="Uses Soundfile library for audio writing; prevents OOM errors on long files.",
                value=False,
                interactive=True,
            )

        output_format = gr.Dropdown(
            label="Output format",
            choices=["WAV", "FLAC", "MP3", "M4A", "OGG", "OPUS", "AIFF", "AC3"],
            value="WAV",
            interactive=True,
        )
        output_bitrate = gr.Textbox(
            label="Output bitrate (e.g. 320k)",
            info="Target bitrate for lossy formats (e.g., 320k, 256k). Leave blank for default.",
            value="",
            placeholder="Leave blank for default",
            interactive=True,
        )
        output_dir = gr.Textbox(
            label="Output directory",
            info="Folder path where separated stem files will be saved.",
            value=audios_folder,
            interactive=True,
        )
        with gr.Row():
            sample_rate = gr.Textbox(
                label="Sample rate",
                info="Output audio sample rate in Hz. Default: 44100.",
                value=44100,
                interactive=True,
            )
            chunk_duration = gr.Textbox(
                label="Chunk duration (s) [0 = disabled]",
                info="Splits long audio into segments to prevent OOM errors. 0 = disabled.",
                value=0,
                interactive=True,
            )
        with gr.Row():
            normalization = gr.Textbox(
                label="Normalization",
                info="Max peak threshold (0-1) to normalize input/output audio and prevent clipping.",
                value=0.9,
                interactive=True,
            )
            amplification = gr.Textbox(
                label="Amplification",
                info="Min peak threshold (0-1) to amplify input/output audio if below this level.",
                value=0.0,
                interactive=True,
            )

    with gr.Tab("VR") as vr_tab:
        vr_model = gr.Dropdown(
            label="Model",
            choices=get_models_by_type("VR"),
            interactive=True,
        )
        with gr.Accordion("Settings", open=False):
            vr_enable_tta = gr.Checkbox(
                label="Enable TTA",
                info="Test-Time-Augmentation. Multiple inference passes for better quality (slower).",
                value=False,
                interactive=True,
            )
            vr_high_end_process = gr.Checkbox(
                label="High-end process",
                info="Mirrors missing high-frequency range in output.",
                value=False,
                interactive=True,
            )
            vr_enable_post_process = gr.Checkbox(
                label="Enable post-process",
                info="Removes residual instrumental artifacts from vocal stems.",
                value=False,
                interactive=True,
            )
            with gr.Row():
                vr_aggression = gr.Slider(
                    label="Aggression",
                    info="Higher = deeper extraction. Default: 5. Values >5 may muddy non-vocal models.",
                    minimum=-100,
                    maximum=100,
                    value=5,
                    interactive=True,
                )
                vr_post_process_threshold = gr.Slider(
                    label="Post-process threshold",
                    info="Higher removes more artifacts but may increase bleed.",
                    minimum=0.1,
                    maximum=0.3,
                    step=0.01,
                    value=0.2,
                    interactive=True,
                )
            with gr.Row():
                vr_batch_size = gr.Textbox(
                    label="Batch size",
                    info="Higher = more RAM usage, slightly faster.",
                    value=4,
                    interactive=True,
                )
                vr_window_size = gr.Dropdown(
                    label="Window size",
                    info="320 = higher quality/slower, 512 = balanced, 1024 = faster/lower quality.",
                    choices=[1024, 512, 320],
                    value=512,
                    interactive=True,
                    allow_custom_value=True,
                )

    with gr.Tab("MDX") as mdx_tab:
        mdx_model = gr.Dropdown(
            label="Model",
            choices=get_models_by_type("MDX"),
            interactive=True,
        )
        with gr.Accordion("Settings", open=False):
            mdx_enable_denoise = gr.Checkbox(
                label="Enable denoise",
                info="Reduces residual noise from separation process.",
                value=False,
                interactive=True,
            )
            mdx_overlap = gr.Slider(
                label="Overlap",
                info="Higher = better quality/slower.",
                minimum=0.001,
                maximum=0.999,
                value=0.25,
                interactive=True,
            )
            with gr.Row():
                mdx_batch_size = gr.Textbox(
                    label="Batch size",
                    info="Higher = more VRAM usage, slightly faster.",
                    value=1,
                    interactive=True,
                )
                mdx_segment_size = gr.Dropdown(
                    label="Segment size",
                    info="Larger = more VRAM, potentially better results.",
                    choices=[128, 256, 512, 1024],
                    value=256,
                    allow_custom_value=True,
                    interactive=True,
                )
                mdx_hop_length = gr.Textbox(
                    label="Hop length",
                    info="Modifies frequency analysis granularity.",
                    value=1024,
                    interactive=True,
                )

    with gr.Tab("MDXC") as mdxc_tab:
        mdxc_model = gr.Dropdown(
            label="Model",
            choices=get_models_by_type("MDXC"),
            interactive=True,
        )
        with gr.Accordion("Settings", open=False):
            mdxc_override_model_segment_size = gr.Checkbox(
                label="Override model segment size",
                info="Forces manual segment size instead of model default.",
                value=False,
                interactive=True,
            )
            mdxc_overlap = gr.Slider(
                label="Overlap",
                info="Higher = better quality/slower.",
                minimum=2,
                maximum=50,
                step=1,
                value=8,
                interactive=True,
            )
            with gr.Row():
                mdxc_batch_size = gr.Textbox(
                    label="Batch size",
                    info="Higher = more VRAM usage, slightly faster.",
                    value=1,
                    interactive=True,
                )
                mdxc_segment_size = gr.Dropdown(
                    label="Segment size",
                    info="Higher captures longer patterns, increases memory.",
                    choices=[128, 256, 512],
                    value=256,
                    allow_custom_value=True,
                    interactive=True,
                )
                mdxc_pitch_shift = gr.Textbox(
                    label="Pitch shift (semitones)",
                    info="Shifts audio pitch by semitones to aid stem detection.",
                    value=0,
                    interactive=True,
                )

    with gr.Tab("Demucs") as demucs_tab:
        demucs_model = gr.Dropdown(
            label="Model",
            choices=get_models_by_type("Demucs"),
            interactive=True,
        )
        with gr.Accordion("Settings", open=False):
            demucs_segments_enabled = gr.Checkbox(
                label="Segments enabled",
                info="Enables segment-wise processing for long audio files.",
                value=True,
                interactive=True,
            )
            demucs_overlap = gr.Slider(
                label="Overlap",
                info="Higher = better quality/slower.",
                minimum=0.001,
                maximum=0.999,
                value=0.25,
                interactive=True,
            )
            with gr.Row():
                demucs_segment_size = gr.Dropdown(
                    label="Segment size",
                    info="Larger = better quality/slower. 'Default' recommended.",
                    choices=["Default", 128, 256],
                    value="Default",
                    allow_custom_value=True,
                    interactive=True,
                )
                demucs_shifts = gr.Textbox(
                    label="Shifts",
                    info="Number of predictions with random shifts. Higher = better quality, significantly slower.",
                    value=2,
                    interactive=True,
                )

    tab_selected = gr.Textbox(
        label="Tab selected",
        value="VR",
        interactive=False,
        visible=False,
    )

    run_uvr_button = gr.Button("Run")
    output_files = gr.File(
        label="Output files",
        file_count="multiple",
        type="filepath",
        interactive=False,
    )

    run_uvr_button.click(
        fn=run_uvr,
        inputs=[
            audio_path,
            audio_upload,
            output_format,
            output_bitrate,
            output_dir,
            invert_spect,
            normalization,
            amplification,
            single_stem,
            sample_rate,
            chunk_duration,
            use_autocast,
            use_soundfile,
            vr_model,
            vr_batch_size,
            vr_window_size,
            vr_aggression,
            vr_enable_tta,
            vr_high_end_process,
            vr_enable_post_process,
            vr_post_process_threshold,
            mdx_model,
            mdx_segment_size,
            mdx_overlap,
            mdx_batch_size,
            mdx_hop_length,
            mdx_enable_denoise,
            mdxc_model,
            mdxc_segment_size,
            mdxc_override_model_segment_size,
            mdxc_overlap,
            mdxc_batch_size,
            mdxc_pitch_shift,
            demucs_model,
            demucs_segment_size,
            demucs_shifts,
            demucs_overlap,
            demucs_segments_enabled,
            tab_selected,
        ],
        outputs=output_files,
    )

    vr_tab.select(lambda: "VR", None, tab_selected)
    mdx_tab.select(lambda: "MDX", None, tab_selected)
    mdxc_tab.select(lambda: "MDXC", None, tab_selected)
    demucs_tab.select(lambda: "Demucs", None, tab_selected)
