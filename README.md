# IR Bilinear Blender Web App

A Streamlit web application for real-time bilinear blending between 4 impulse response files with convolution processing.

## Features

- **2D IR Blending**: Interactive sliders to blend between 4 IR files positioned at corners
- **Real-time Convolution**: Apply blended IR to input audio with live playback
- **Visual Feedback**: 
  - 2D position visualization
  - Waveform displays
  - Spectrograms
  - Blend weight indicators
- **Audio Controls**:
  - Dry/Wet mixing
  - Output gain control
  - Side-by-side audio comparison

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your files:**
   - Create an `IR` folder in the same directory
   - Place exactly 4 `.wav` IR files in the folder
   - Optionally, add an input audio file (e.g., `input_audio.wav`)

3. **Run the app:**
   ```bash
   streamlit run ir_blender_app.py
   ```

## File Structure
```
project/
├── ir_blender_app.py
├── requirements.txt
├── README.md
├── IR/
│   ├── reverb1.wav
│   ├── reverb2.wav
│   ├── reverb3.wav
│   └── reverb4.wav
└── input_audio.wav (optional)
```

## IR File Positioning

The app positions your 4 IR files as:
- **Bottom-left (0,0)**: First IR file
- **Bottom-right (1,0)**: Second IR file  
- **Top-left (0,1)**: Third IR file
- **Top-right (1,1)**: Fourth IR file

## How to Use

1. **Load the app** - IR files are automatically loaded from the `IR` folder
2. **Adjust X/Y sliders** - Blend between the 4 IR files in 2D space
3. **Set input audio path** - Enter your input audio filename
4. **Configure processing** - Adjust dry/wet mix and output gain
5. **Listen and compare** - Use the audio players to hear results

## Benefits over Jupyter Notebooks

- ✅ **Better slider performance** - Only recalculates when you release the slider
- ✅ **Clean web interface** - Professional layout with organized controls
- ✅ **Easy sharing** - Send URL to others to use your app
- ✅ **Caching** - Faster performance with automatic data caching
- ✅ **Responsive design** - Works on different screen sizes

## Troubleshooting

- **"No IR files loaded"**: Ensure you have an `IR` folder with 4 `.wav` files
- **"No input audio loaded"**: Check the file path in the sidebar input
- **Audio not playing**: Some browsers may block autoplay - click the play button manually
