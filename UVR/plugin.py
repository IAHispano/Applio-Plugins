import os, sys
import json
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from tabs.plugins.installed.UVR.uvr import Separator

plugin_folder = os.path.relpath(
    os.path.join(now_dir, "tabs", "plugins", "installed", "UVR")
)


def get_models_by_type(type):
    download_checks_path = os.path.join(plugin_folder, "models", "download_checks.json")

    model_downloads_list = json.load(open(download_checks_path, encoding="utf-8"))

    filtered_demucs_v4 = {
        key: value
        for key, value in model_downloads_list["demucs_download_list"].items()
        if key.startswith("Demucs v4")
    }

    model_files_grouped_by_type = {
        "VR": model_downloads_list["vr_download_list"],
        "MDX": {
            **model_downloads_list["mdx_download_list"],
            **model_downloads_list["mdx_download_vip_list"],
        },
        "Demucs": filtered_demucs_v4,
        "MDXC": {
            **model_downloads_list["mdx23c_download_list"],
            **model_downloads_list["mdx23c_download_vip_list"],
            **model_downloads_list["roformer_download_list"],
        },
    }

    results = []
    for model_name, model_info in model_files_grouped_by_type[type].items():
        results.append(model_info)

    return results


def run_uvr(
    audio,
    output_format,
    output_dir,
    invert_spect,
    normalization,
    single_stem,
    sample_rate,
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
    # demucs_model,
    # demucs_segment_size,
    # demucs_shifts,
    # demucs_overlap,
    # demucs_segments_enabled,
    tab_selected,
):
    if tab_selected == "VR":
        model = vr_model
    elif tab_selected == "MDX":
        model = mdx_model
    elif tab_selected == "MDXC":
        model = mdxc_model
    # elif tab_selected == "Demucs":
    #     model = demucs_model

    if single_stem == "None":
        single_stem = None

    separator = Separator(
        model_file_dir=os.path.join(plugin_folder, "models"),
        output_dir=output_dir,
        output_format=output_format,
        normalization_threshold=float(normalization),
        output_single_stem=single_stem,
        invert_using_spec=invert_spect,
        sample_rate=int(sample_rate),
        mdx_params={
            "hop_length": int(mdx_hop_length),
            "segment_size": int(mdx_segment_size),
            "overlap": float(mdx_overlap),
            "batch_size": int(mdx_batch_size),
            "enable_denoise": mdx_enable_denoise,
        },
        vr_params={
            "batch_size": int(vr_batch_size),
            "window_size": int(vr_window_size),
            "aggression": int(vr_aggression),
            "enable_tta": vr_enable_tta,
            "enable_post_process": vr_enable_post_process,
            "post_process_threshold": float(vr_post_process_threshold),
            "high_end_process": vr_high_end_process,
        },
        mdxc_params={
            "segment_size": int(mdxc_segment_size),
            "batch_size": int(mdxc_batch_size),
            "overlap": int(mdxc_overlap),
            "override_model_segment_size": mdxc_override_model_segment_size,
            "pitch_shift": int(mdxc_pitch_shift),
        },
    )
    """
    demucs_params={
        "segment_size": demucs_segment_size,
        "shifts": demucs_shifts,
        "overlap": demucs_overlap,
        "segments_enabled": demucs_segments_enabled,
    },
    """
    separator.load_model(model_filename=model)

    results = []
    files = separator.separate(audio)
    try:
        for file in files:
            file_path = os.path.join(output_dir, file)
            results.append(file_path)
        return results
    except AttributeError:
        return os.path.join(output_dir, files)


def applio_plugin():
    audio = gr.Audio(
        label="Input audio",
        sources=["upload", "microphone"],
        type="filepath",
        interactive=True,
    )

    single_stem = gr.Radio(
        label="Single stem",
        choices=[
            "None",
            "Instrumental",
            "Vocals",
            "Drums",
            "Bass",
            "Guitar",
            "Piano",
            "Other",
        ],
        value="None",
        interactive=True,
    )

    with gr.Accordion("Advanced Settings", open=False):
        invert_spect = gr.Checkbox(
            label="Invert spectrogram",
            value=False,
            interactive=True,
        )

        output_format = gr.Radio(
            label="Output format",
            choices=["wav", "mp3"],
            value="wav",
            interactive=True,
        )

        output_dir = gr.Textbox(
            label="Output directory",
            value=os.path.join(plugin_folder, "output"),
            interactive=True,
        )

        with gr.Row():
            sample_rate = gr.Textbox(
                label="Sample rate",
                value=44100,
                interactive=True,
            )

            normalization = gr.Textbox(
                label="Normalization",
                value=0.9,
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
                value=False,
                interactive=True,
            )
            vr_high_end_process = gr.Checkbox(
                label="High-end process",
                value=False,
                interactive=True,
            )
            vr_enable_post_process = gr.Checkbox(
                label="Enable post-process",
                value=False,
                interactive=True,
            )
            with gr.Row():
                vr_aggression = gr.Slider(
                    label="Aggression",
                    minimum=-100,
                    maximum=100,
                    value=5,
                    interactive=True,
                )
                vr_post_process_threshold = gr.Slider(
                    label="Post-process threshold",
                    minimum=0.1,
                    maximum=0.3,
                    step=0.01,
                    value=0.2,
                    interactive=True,
                )
            with gr.Row():
                vr_batch_size = gr.Textbox(
                    label="Batch size",
                    value=4,
                    interactive=True,
                )
                vr_window_size = gr.Dropdown(
                    label="Window size",
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
                value=False,
                interactive=True,
            )
            with gr.Row():
                mdx_overlap = gr.Slider(
                    label="Overlap",
                    minimum=0.001,
                    maximum=0.999,
                    value=0.25,
                    interactive=True,
                )
            with gr.Row():
                mdx_batch_size = gr.Textbox(
                    label="Batch size",
                    value=1,
                    interactive=True,
                )
                mdx_segment_size = gr.Textbox(
                    label="Segment size",
                    value=256,
                    interactive=True,
                )
                mdx_hop_length = gr.Textbox(
                    label="Hop length",
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
                value=False,
            )
            with gr.Row():
                mdxc_overlap = gr.Slider(
                    label="Overlap",
                    minimum=0.001,
                    maximum=0.999,
                    value=0.25,
                    interactive=True,
                )
            with gr.Row():
                mdxc_batch_size = gr.Textbox(
                    label="Batch size",
                    value=1,
                    interactive=True,
                )
                mdxc_segment_size = gr.Textbox(
                    label="Segment size",
                    value=256,
                    interactive=True,
                )
                mdxc_pitch_shift = gr.Textbox(
                    label="Hop length",
                    value=0,
                    interactive=True,
                )

    with gr.Tab("Demucs") as demucs_tab:
        gr.Markdown("Demucs is not available in this version of the plugin.")
        """
        demucs_model = gr.Dropdown(
            label="Model",
            choices=get_models_by_type("Demucs"),
            interactive=True,
        )
        with gr.Accordion("Settings", open=False):
            demucs_segments_enabled = gr.Checkbox(
                label="Segments enabled",
                value=True,
                interactive=True,
            )    
            demucs_overlap = gr.Slider(
                label="Overlap",
                minimum=0.001,
                maximum=0.999,
                value=0.25,
                interactive=True,
            )
            with gr.Row():
                demucs_segment_size = gr.Textbox(
                    label="Segment size",
                    value="Default",
                    interactive=True,
                )
                demucs_shifts = gr.Textbox(
                    label="Shifts",
                    value=2,
                    interactive=True,
                )
        """

    tab_selected = gr.Textbox(
        label="Tab selected",
        value="VR",
        interactive=False,
        visible=False,
    )

    run_uvr_button = gr.Button("Run")
    output_files = gr.File(
        label="Output files", file_count="multiple", type="filepath", interactive=False
    )

    run_uvr_button.click(
        fn=run_uvr,
        inputs=[
            audio,
            output_format,
            output_dir,
            invert_spect,
            normalization,
            single_stem,
            sample_rate,
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
            # demucs_model,
            # demucs_segment_size,
            # demucs_shifts,
            # demucs_overlap,
            # demucs_segments_enabled,
            tab_selected,
        ],
        outputs=output_files,
    )

    vr_tab.select(lambda: "VR", None, tab_selected)
    mdx_tab.select(lambda: "MDX", None, tab_selected)
    mdxc_tab.select(lambda: "MDXC", None, tab_selected)
    demucs_tab.select(lambda: "Demucs", None, tab_selected)
