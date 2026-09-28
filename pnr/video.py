"""Portable dashboard playback alongside the original downloadable MP4."""
from pathlib import Path
import subprocess


def webm_copy(mp4_path):
    import imageio_ffmpeg
    source = Path(mp4_path)
    target = source.with_suffix('.webm')
    temporary = target.with_name(target.stem+'.pending.webm')
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y',
            '-hide_banner', '-loglevel', 'error', '-i', str(source), '-an',
            '-c:v', 'libvpx-vp9', '-crf', '32', '-b:v', '0', '-row-mt', '1',
            '-cpu-used', '4', '-threads', '4', '-pix_fmt', 'yuv420p', str(temporary)],
            check=True, capture_output=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
