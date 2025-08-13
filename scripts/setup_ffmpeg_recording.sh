#!/bin/bash
# TMRT Demo Recording with FFmpeg Only
# Professional competition demo recording using native Kali Linux ffmpeg

echo "🎥 TMRT Competition Demo Recording - FFmpeg Only"
echo "==============================================="
echo "Setting up professional demo recording using native ffmpeg"

# Check ffmpeg installation
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 Installing ffmpeg..."
    sudo apt update && sudo apt install -y ffmpeg
else
    echo "✅ FFmpeg already installed"
fi

# Create recording directory
RECORDING_DIR="$HOME/tmrt_demo_recording"
mkdir -p "$RECORDING_DIR"
cd "$RECORDING_DIR"

echo "📁 Recording directory: $RECORDING_DIR"

# Get display information
DISPLAY_INFO=$(xrandr | grep ' connected' | head -1)
echo "🖥️ Display: $DISPLAY_INFO"

# Create recording script
cat > start_recording.sh << 'EOF'
#!/bin/bash
# Start FFmpeg screen recording with audio

echo "🎬 Starting TMRT Demo Recording..."
echo "Press Ctrl+C to stop recording"

# Get screen resolution
RESOLUTION=$(xrandr | grep '\*' | head -1 | awk '{print $1}')
echo "📺 Recording at resolution: $RESOLUTION"

# Start recording with ffmpeg
ffmpeg \
  -f x11grab \
  -r 30 \
  -s $RESOLUTION \
  -i :0.0 \
  -f pulse \
  -ac 2 \
  -i default \
  -c:v libx264 \
  -preset fast \
  -crf 18 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  tmrt_demo_$(date +%Y%m%d_%H%M%S).mp4

echo "✅ Recording saved as: tmrt_demo_$(date +%Y%m%d_%H%M%S).mp4"
EOF

chmod +x start_recording.sh

# Create recording with region selection
cat > record_region.sh << 'EOF'
#!/bin/bash
# Record specific region of screen

echo "🎯 Select recording region with mouse"
echo "Click and drag to select the area to record"

# Use xwininfo to let user select region
echo "Click on the window or area you want to record..."
WINDOW_INFO=$(xwininfo)
WINDOW_ID=$(echo "$WINDOW_INFO" | grep "Window id:" | awk '{print $4}')
GEOMETRY=$(echo "$WINDOW_INFO" | grep "geometry" | awk '{print $2}' | cut -d'+' -f1)
X_OFFSET=$(echo "$WINDOW_INFO" | grep "Absolute upper-left X:" | awk '{print $4}')
Y_OFFSET=$(echo "$WINDOW_INFO" | grep "Absolute upper-left Y:" | awk '{print $4}')

echo "📐 Recording region: ${GEOMETRY} at offset ${X_OFFSET},${Y_OFFSET}"

ffmpeg \
  -f x11grab \
  -r 30 \
  -s $GEOMETRY \
  -i :0.0+${X_OFFSET},${Y_OFFSET} \
  -f pulse \
  -ac 2 \
  -i default \
  -c:v libx264 \
  -preset fast \
  -crf 18 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  tmrt_region_demo_$(date +%Y%m%d_%H%M%S).mp4

echo "✅ Region recording completed!"
EOF

chmod +x record_region.sh

# Create terminal recording script
cat > record_terminal.sh << 'EOF'
#!/bin/bash
# Record terminal session with ffmpeg

echo "💻 Recording Terminal Session"

# Find terminal window
TERMINAL_WINDOW=$(xdotool search --onlyvisible --class "terminal\|gnome-terminal\|xterm\|konsole" | head -1)

if [ -n "$TERMINAL_WINDOW" ]; then
    # Get terminal window geometry
    eval $(xdotool getwindowgeometry --shell $TERMINAL_WINDOW)
    echo "📐 Terminal window: ${WIDTH}x${HEIGHT} at ${X},${Y}"
    
    ffmpeg \
      -f x11grab \
      -r 30 \
      -s ${WIDTH}x${HEIGHT} \
      -i :0.0+${X},${Y} \
      -f pulse \
      -ac 2 \
      -i default \
      -c:v libx264 \
      -preset fast \
      -crf 18 \
      -c:a aac \
      -b:a 128k \
      -movflags +faststart \
      tmrt_terminal_$(date +%Y%m%d_%H%M%S).mp4
else
    echo "❌ No terminal window found. Please open a terminal first."
fi
EOF

chmod +x record_terminal.sh

# Create demo recording sequence script
cat > record_demo_sequence.sh << 'EOF'
#!/bin/bash
# Complete TMRT demo recording sequence

echo "🎬 TMRT Competition Demo Recording Sequence"
echo "=========================================="

# Demo segments
SEGMENTS=(
    "intro:30:Introduction and framework overview"
    "live_demo:240:Live vulnerability discovery demonstration" 
    "results:90:Results overview and findings summary"
    "technical:60:Technical architecture and reproducibility"
    "conclusion:30:Conclusion and competition readiness"
)

mkdir -p segments

echo "📋 Demo will be recorded in 5 segments:"
for segment in "${SEGMENTS[@]}"; do
    IFS=':' read -r name duration description <<< "$segment"
    echo "  • $name ($duration seconds): $description"
done

echo ""
echo "🎯 Ready to record each segment?"
read -p "Press Enter to start with the introduction segment..."

# Record each segment
for segment in "${SEGMENTS[@]}"; do
    IFS=':' read -r name duration description <<< "$segment"
    
    echo ""
    echo "🎬 Recording: $description"
    echo "Duration: $duration seconds"
    echo "Prepare your content and press Enter when ready..."
    read
    
    # Start countdown
    echo "Starting in..."
    for i in 3 2 1; do
        echo "$i..."
        sleep 1
    done
    echo "🔴 RECORDING NOW!"
    
    # Record segment
    timeout ${duration} ffmpeg \
      -f x11grab \
      -r 30 \
      -s $(xrandr | grep '\*' | head -1 | awk '{print $1}') \
      -i :0.0 \
      -f pulse \
      -ac 2 \
      -i default \
      -c:v libx264 \
      -preset fast \
      -crf 18 \
      -c:a aac \
      -b:a 128k \
      -movflags +faststart \
      segments/${name}_segment.mp4 2>/dev/null
    
    echo "✅ Segment '$name' recorded!"
    echo "Saved as: segments/${name}_segment.mp4"
done

echo ""
echo "🎞️ All segments recorded! Now combining into final demo..."

# Combine segments
cat > segments/concat_list.txt << CONCAT_EOF
file 'intro_segment.mp4'
file 'live_demo_segment.mp4'
file 'results_segment.mp4'
file 'technical_segment.mp4'
file 'conclusion_segment.mp4'
CONCAT_EOF

ffmpeg -f concat -safe 0 -i segments/concat_list.txt -c copy tmrt_complete_demo_$(date +%Y%m%d_%H%M%S).mp4

echo "🏆 Complete demo video created!"
echo "📁 Location: $(pwd)/tmrt_complete_demo_$(date +%Y%m%d_%H%M%S).mp4"
EOF

chmod +x record_demo_sequence.sh

# Create simple one-shot recording
cat > quick_record.sh << 'EOF'
#!/bin/bash
# Quick one-shot demo recording

echo "🎬 Quick TMRT Demo Recording"
echo "==========================="
echo "This will record a complete 5-7 minute demo"
echo "Make sure your terminal and editor are ready"
echo ""
echo "📋 Demo outline:"
echo "1. Introduction (30 seconds)"
echo "2. Live demonstration (4 minutes)" 
echo "3. Results showcase (2 minutes)"
echo "4. Conclusion (30 seconds)"
echo ""

read -p "Ready to start recording? Press Enter..."

# Countdown
echo "Starting in..."
for i in 5 4 3 2 1; do
    echo "$i..."
    sleep 1
done

echo "🔴 RECORDING NOW! (7 minutes max)"

# Record 7-minute demo
timeout 420 ffmpeg \
  -f x11grab \
  -r 30 \
  -s $(xrandr | grep '\*' | head -1 | awk '{print $1}') \
  -i :0.0 \
  -f pulse \
  -ac 2 \
  -i default \
  -c:v libx264 \
  -preset fast \
  -crf 18 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  tmrt_demo_final_$(date +%Y%m%d_%H%M%S).mp4

echo ""
echo "✅ Recording complete!"
echo "📁 Demo saved as: tmrt_demo_final_$(date +%Y%m%d_%H%M%S).mp4"
echo "📊 File size: $(ls -lh tmrt_demo_final_*.mp4 | tail -1 | awk '{print $5}')"
EOF

chmod +x quick_record.sh

# Create demo preparation script
cat > prepare_demo.sh << 'EOF'
#!/bin/bash
# Prepare environment for demo recording

echo "⚙️ Preparing TMRT Demo Environment"
echo "================================="

# Navigate to project directory
cd /home/o1/Desktop/openai-res

# Set up terminal
echo "🖥️ Setting up terminal..."
export PS1="tmrt-demo:$ "
clear

# Open key files for demo
echo "📄 Preparing key files..."
echo "Files to have open during demo:"
echo "• COMPETITION_WRITEUP.md (overview)"
echo "• src/tmrt/ (source code)"
echo "• FINAL_SUBMISSION_PACKAGE/ (results)"
echo "• kaggle_submission.json (findings)"

# Create demo script
cat > demo_talking_points.md << 'DEMO_EOF'
# TMRT Demo Script - 5-7 Minutes

## 1. Introduction (30 seconds)
"Welcome to the Token-Manifold Red Teaming framework demonstration for the OpenAI GPT-OSS-20b Red Teaming Challenge. I'm about to show you how our advanced automated framework discovered 10 novel vulnerabilities with an 87.8% average novelty score."

## 2. Live Demonstration (4 minutes)
- Show project structure: `ls -la`
- Display framework components: `ls src/tmrt/`
- Show key findings: `cat FINAL_SUBMISSION_PACKAGE/submission_files/kaggle_submission.json`
- Demonstrate configuration: `cat configs/production.yaml`
- Show validation results: `python scripts/final_submission_validator.py`

## 3. Results Showcase (2 minutes)  
- Display findings summary table
- Highlight 6 critical vulnerabilities
- Show novelty scores and categories
- Demonstrate GitHub repository

## 4. Technical Excellence (1 minute)
- Show Docker setup: `cat docker/Dockerfile`
- Display documentation: `ls technical_documentation/`
- Show test coverage and validation

## 5. Conclusion (30 seconds)
"The TMRT framework represents breakthrough research in automated vulnerability discovery. Our submission is ready for the competition with comprehensive documentation, full reproducibility, and significant research contributions to AI safety."

DEMO_EOF

echo "📋 Demo script created: demo_talking_points.md"
echo ""
echo "✅ Demo environment prepared!"
echo "🎬 Ready to record with any of these options:"
echo "  • ./quick_record.sh - Single 7-minute recording"
echo "  • ./record_demo_sequence.sh - Segmented recording"
echo "  • ./start_recording.sh - Manual start/stop"
EOF

chmod +x prepare_demo.sh

echo ""
echo "🏆 FFmpeg Recording Setup Complete!"
echo "===================================="
echo "📁 Recording directory: $RECORDING_DIR"
echo ""
echo "🎬 Recording options available:"
echo "  • ./quick_record.sh - Quick 7-minute demo (RECOMMENDED)"
echo "  • ./record_demo_sequence.sh - Professional segmented recording"
echo "  • ./start_recording.sh - Full screen recording"
echo "  • ./record_region.sh - Record specific window/region"
echo "  • ./record_terminal.sh - Terminal-focused recording"
echo ""
echo "⚙️ First run: ./prepare_demo.sh"
echo "🎯 Then choose your recording method!"
echo ""
echo "🔧 FFmpeg features enabled:"
echo "  ✅ Full screen recording (1080p+)"
echo "  ✅ Audio capture (microphone + system)"
echo "  ✅ High quality H.264 encoding"
echo "  ✅ Fast preset for real-time recording"
echo "  ✅ Professional MP4 output"
echo ""
echo "🚀 Ready to create winning competition demo!"
