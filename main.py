import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from scipy.signal import fftconvolve
from collections import namedtuple
import io
import base64

st.set_page_config(
    page_title="IR Bilinear Blender",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Impulse Response Bilinear Blender")
st.markdown("Interactive 2D blending between 4 impulse response files with real-time convolution")

IR_file = namedtuple('AudioFile', ['filename', 'audio_data', 'sample_rate'])

@st.cache_data
def load_ir_files(ir_folder_path='IR'):
    """Load IR files with caching"""
    ir_files = []
    
    if not os.path.exists(ir_folder_path):
        st.error(f"IR folder '{ir_folder_path}' not found!")
        return []
    
    for filename in os.listdir(ir_folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(ir_folder_path, filename)
            try:
                y, sr = librosa.load(file_path, mono=True, sr=48000)
                audio_file = IR_file(filename=filename, audio_data=y, sample_rate=sr)
                ir_files.append(audio_file)
                st.success(f'✅ Loaded {filename} ({len(y)} samples at {sr} Hz)')
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {e}")
    
    if len(ir_files) != 4:
        st.warning(f"Expected 4 IR files, found {len(ir_files)}. Padding with zeros.")
        while len(ir_files) < 4:
            empty_ir = IR_file(
                filename=f"empty_{len(ir_files)}.wav", 
                audio_data=np.zeros(48000), 
                sample_rate=48000
            )
            ir_files.append(empty_ir)
    
    return ir_files[:4]

@st.cache_data
def load_input_audio(file_path, target_sr=48000):
    """Load input audio with caching"""
    try:
        if os.path.exists(file_path):
            audio, sr = librosa.load(file_path, mono=True, sr=target_sr)
            return audio, sr
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading input audio: {e}")
        return None, None

def bilinear_interpolation(x, y, ir_files):
    """Perform bilinear interpolation between 4 IR files"""
    ir_00 = ir_files[0].audio_data  # bottom-left
    ir_10 = ir_files[1].audio_data  # bottom-right
    ir_01 = ir_files[2].audio_data  # top-left
    ir_11 = ir_files[3].audio_data  # top-right
    
    max_length = max(len(ir_00), len(ir_10), len(ir_01), len(ir_11))
    
    def pad_to_length(audio, target_length):
        if len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)), 'constant')
        return audio[:target_length]
    
    ir_00 = pad_to_length(ir_00, max_length)
    ir_10 = pad_to_length(ir_10, max_length)
    ir_01 = pad_to_length(ir_01, max_length)
    ir_11 = pad_to_length(ir_11, max_length)
    
    ir_bottom = ir_00 * (1 - x) + ir_10 * x
    ir_top = ir_01 * (1 - x) + ir_11 * x
    blended_ir = ir_bottom * (1 - y) + ir_top * y
    
    return blended_ir

def create_position_plot(x_coord, y_coord, ir_files):
    """Create 2D position visualization"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    ax.scatter([0, 1, 0, 1], [0, 0, 1, 1], s=300, c='lightblue', alpha=0.7, edgecolors='black')
    ax.text(0, 0, ir_files[0].filename, ha='center', va='top', fontsize=10, weight='bold')
    ax.text(1, 0, ir_files[1].filename, ha='center', va='top', fontsize=10, weight='bold')
    ax.text(0, 1, ir_files[2].filename, ha='center', va='bottom', fontsize=10, weight='bold')
    ax.text(1, 1, ir_files[3].filename, ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax.scatter(x_coord, y_coord, s=400, c='red', marker='x', linewidths=4)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('2D Blend Position', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def create_audio_plots(blended_ir, ir_files, input_audio=None, final_audio=None, x_coord=0, y_coord=0):
    """Create audio waveform and spectrogram plots"""
    if input_audio is not None and final_audio is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
        
        time_input = np.arange(len(input_audio)) / 48000
        ax1.plot(time_input, input_audio, 'g-', linewidth=1)
        ax1.set_title('Input Audio (Dry)', fontsize=12, weight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        time_ir = np.arange(len(blended_ir)) / ir_files[0].sample_rate
        ax2.plot(time_ir, blended_ir, 'b-', linewidth=1)
        ax2.set_title(f'Blended IR ({x_coord:.2f}, {y_coord:.2f})', fontsize=12, weight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        time_output = np.arange(len(final_audio)) / 48000
        ax3.plot(time_output, final_audio, 'r-', linewidth=1)
        ax3.set_title('Convolved Output', fontsize=12, weight='bold')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True, alpha=0.3)
        
        D = librosa.stft(blended_ir)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax4)
        ax4.set_title('IR Spectrogram', fontsize=12, weight='bold')
        
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        time_ir = np.arange(len(blended_ir)) / ir_files[0].sample_rate
        ax1.plot(time_ir, blended_ir, 'b-', linewidth=1)
        ax1.set_title(f'Blended IR - Position: ({x_coord:.2f}, {y_coord:.2f})', fontsize=12, weight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        D = librosa.stft(blended_ir)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax2)
        ax2.set_title('IR Spectrogram', fontsize=12, weight='bold')
    
    plt.tight_layout()
    return fig

def audio_to_base64(audio_array, sample_rate):
    """Convert audio array to base64 for HTML audio player"""

    audio_normalized = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
    
    audio_int16 = (audio_normalized * 32767).astype(np.int16)
    
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
    return audio_base64

def main():
    with st.spinner("Loading IR files..."):
        ir_files = load_ir_files()
    
    if not ir_files:
        st.error("No IR files loaded. Please ensure you have an 'IR' folder with .wav files.")
        return
    
    st.sidebar.header("🎛️ Blend Controls")
    
    x_coord = st.sidebar.slider("X Position", 0.0, 1.0, 0.5, 0.01, help="Left-Right blending")
    y_coord = st.sidebar.slider("Y Position", 0.0, 1.0, 0.5, 0.01, help="Bottom-Top blending")
    
    weight_00 = (1 - x_coord) * (1 - y_coord)
    weight_10 = x_coord * (1 - y_coord)
    weight_01 = (1 - x_coord) * y_coord
    weight_11 = x_coord * y_coord
    
    st.sidebar.subheader("📍 IR File Positions")
    st.sidebar.write(f"**Bottom-left (0,0):** {ir_files[0].filename} - {weight_00:.3f}")
    st.sidebar.write(f"**Bottom-right (1,0):** {ir_files[1].filename} - {weight_10:.3f}")
    st.sidebar.write(f"**Top-left (0,1):** {ir_files[2].filename} - {weight_01:.3f}")
    st.sidebar.write(f"**Top-right (1,1):** {ir_files[3].filename} - {weight_11:.3f}")
    
    st.sidebar.header("🎤 Input Audio")
    input_audio_path = st.sidebar.text_input("Input Audio Path", "input_audio.wav")
    
    input_audio, input_sr = load_input_audio(input_audio_path)
    
    if input_audio is not None:
        st.sidebar.success(f"✅ Input audio loaded: {len(input_audio)/input_sr:.2f}s")
        
        st.sidebar.subheader("🔧 Convolution Settings")
        dry_wet_mix = st.sidebar.slider("Dry/Wet Mix", 0.0, 1.0, 1.0, 0.01, 
                                       help="0.0 = dry only, 1.0 = wet only")
        output_gain = st.sidebar.slider("Output Gain", 0.0, 2.0, 0.5, 0.01)
    else:
        st.sidebar.warning("❌ No input audio loaded")
    
    blended_ir = bilinear_interpolation(x_coord, y_coord, ir_files)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Blend Position")
        position_fig = create_position_plot(x_coord, y_coord, ir_files)
        st.pyplot(position_fig)
        plt.close()
    
    with col2:
        if input_audio is not None:
            with st.spinner("Processing audio..."):
                convolved_audio = fftconvolve(input_audio, blended_ir, mode='full')
                
                if np.max(np.abs(convolved_audio)) > 0:
                    convolved_audio = convolved_audio / np.max(np.abs(convolved_audio))
                
                dry_padded = np.pad(input_audio, (0, max(0, len(convolved_audio) - len(input_audio))), 'constant')
                wet_padded = np.pad(convolved_audio, (0, max(0, len(dry_padded) - len(convolved_audio))), 'constant')
                mixed_audio = (1 - dry_wet_mix) * dry_padded[:len(wet_padded)] + dry_wet_mix * wet_padded
                final_audio = mixed_audio * output_gain
            
            st.subheader("📊 Audio Analysis")
            audio_fig = create_audio_plots(blended_ir, ir_files, input_audio, final_audio, x_coord, y_coord)
            st.pyplot(audio_fig)
            plt.close()
            
            st.subheader("🎵 Audio Playback")
            
            col_dry, col_wet = st.columns(2)
            
            with col_dry:
                st.write("**Original (Dry):**")
                dry_b64 = audio_to_base64(input_audio, input_sr)
                st.markdown(f"""
                <audio controls>
                    <source src="data:audio/wav;base64,{dry_b64}" type="audio/wav">
                </audio>
                """, unsafe_allow_html=True)
            
            with col_wet:
                st.write("**Processed (Wet):**")
                wet_b64 = audio_to_base64(final_audio, input_sr)
                st.markdown(f"""
                <audio controls>
                    <source src="data:audio/wav;base64,{wet_b64}" type="audio/wav">
                </audio>
                """, unsafe_allow_html=True)
        
        else:
            st.subheader("📊 IR Analysis")
            ir_fig = create_audio_plots(blended_ir, ir_files, None, None, x_coord, y_coord)
            st.pyplot(ir_fig)
            plt.close()
            
            st.subheader("🎵 Blended IR Playback")
            ir_b64 = audio_to_base64(blended_ir, ir_files[0].sample_rate)
            st.markdown(f"""
            <audio controls>
                <source src="data:audio/wav;base64,{ir_b64}" type="audio/wav">
            </audio>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
