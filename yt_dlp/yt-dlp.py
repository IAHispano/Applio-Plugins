import gradio as gr
import yt_dlp
import os

def download_wav(url, output_dir=".", debug=False):
    """Downloads a WAV file from YouTube using yt-dlp.

    Args:
        url: The URL of the YouTube video.
        output_dir: The directory to save the WAV file. Defaults to the current directory.
        debug: If True, enable yt-dlp's debug mode. Defaults to False

    Returns:
        The path to the downloaded WAV file or None if the download fails.
        Also prints informative messages during the process.
    """

    ydl_opts = {
        'format': 'bestaudio/best',  # Try to get the best audio quality
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'), # Use the video title
        'noplaylist': True,  # Download only a single video, not a playlist
        'extract_audio': True,  # Extract audio only
        'audio_format': 'wav',  # Convert to WAV format
        'postprocessors': [{  # Use ffmpeg for audio conversion
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'logger': MyLogger() if debug else None # Custom logger for debugging
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            # Get the title from the info_dict to use as the file name
            video_title = info_dict.get('title', 'video')  # Use a default title if it's not found
            file_path = os.path.join(output_dir, f"{video_title}.wav")
            
            if debug:
                print(f"Download complete: {file_path}")
            return file_path
    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading: {e}")
        return None
    except Exception as e:  # Catch other potential errors during download
        print(f"An unexpected error occurred: {e}")
        return None
    
class MyLogger:
    def debug(self, msg):
        print(f"DEBUG: {msg}")
    def warning(self, msg):
        print(f"WARNING: {msg}")
    def error(self, msg):
        print(f"ERROR: {msg}")




def applio_plugin():
    with gr.Row():
        text_inpt  = gr.Textbox(label="URL INPUT", info="inpur your audio/video URL.")
        debug = gr.Checkbox(label="Debug",info="Apply Debbug.",visible=True,value=False,interactive=True)
    cust_dir  = gr.Textbox(label="Output directory", info="paste your custom Output directory.")
    with gr.Row():
        export_button = gr.Button("Download URL")
    with gr.Row():
        aud_opt gr.Audio(label="Output Audio",info="Your output Audio.",visible=True,value=False,interactive=False)

    export_button.click(fn=download_wav, inputs=[text_inpt,debug,cust_dir], outputs=[aud_opt])
    gr.Markdown(
        value=
            "this plugins is for Applio ."
    )