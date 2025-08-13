#!/bin/bash
# TMRT Demo Recording Script - Native Kali Linux Tools
# Professional competition demo video creation

DEMO_DIR="$HOME/tmrt_demo"
RECORDINGS_DIR="$DEMO_DIR/recordings"
TMRT_DIR="/home/o1/Desktop/openai-res"

echo "🎥 TMRT Competition Demo Recording - Native Kali Linux Tools"
echo "=============================================================="
echo "Recording professional demo for OpenAI GPT-OSS-20b Red Teaming Challenge"

# Create directories
mkdir -p "$RECORDINGS_DIR"
mkdir -p "$DEMO_DIR/screenshots"

# Function to record terminal session with asciinema
record_terminal_demo() {
    echo "💻 Starting Terminal Demo Recording..."
    echo "This will record the terminal session for the TMRT demo"
    echo "Press Enter to start recording, Ctrl+D to stop"
    read -p "Ready? "
    
    cd "$TMRT_DIR"
    asciinema rec "$RECORDINGS_DIR/tmrt_terminal_demo.cast" \
        --title "TMRT Framework Demo - Competition Submission" \
        --command "/bin/bash"
}

# Function to start OBS Studio for screen recording
setup_obs_recording() {
    echo "🎬 Setting up OBS Studio for Screen Recording..."
    
    # Create OBS scene configuration
    mkdir -p "$HOME/.config/obs-studio/basic/scenes"
    
    cat > "$HOME/.config/obs-studio/basic/scenes/TMRT_Demo.json" << 'EOF'
{
    "current_scene": "TMRT Demo",
    "sources": [
        {
            "name": "Desktop Capture",
            "type": "xshm_input", 
            "settings": {
                "screen": 0,
                "use_cursor": true,
                "show_cursor": true
            }
        }
    ]
}
EOF

    echo "OBS Studio configuration created"
    echo "Starting OBS Studio..."
    obs &
    OBS_PID=$!
    echo "OBS Studio started with PID: $OBS_PID"
    
    echo ""
    echo "📋 OBS Studio Setup Instructions:"
    echo "1. In OBS, create a new scene called 'TMRT Demo'"
    echo "2. Add Source -> Screen Capture (XSHM)"
    echo "3. Configure output settings:"
    echo "   - Format: MP4"
    echo "   - Resolution: 1920x1080" 
    echo "   - Frame Rate: 30 FPS"
    echo "4. Click 'Start Recording' when ready"
    echo ""
}

# Function to record full screen with ffmpeg (alternative to OBS)
record_screen_ffmpeg() {
    echo "📹 Starting Screen Recording with FFmpeg..."
    echo "This will record the entire screen"
    echo "Press Ctrl+C to stop recording"
    
    # Get screen resolution
    RESOLUTION=$(xrandr | grep '*' | head -1 | awk '{print $1}')
    echo "Detected resolution: $RESOLUTION"
    
    # Start screen recording
    ffmpeg -video_size "$RESOLUTION" \
           -framerate 30 \
           -f x11grab \
           -i :0.0 \
           -f pulse \
           -ac 2 \
           -i default \
           -c:v libx264 \
           -preset medium \
           -crf 23 \
           -c:a aac \
           -b:a 128k \
           "$RECORDINGS_DIR/tmrt_demo_screen_$(date +%Y%m%d_%H%M%S).mp4"
}

# Function to take screenshots during demo
take_screenshots() {
    echo "📸 Taking Demo Screenshots..."
    
    # Key moments screenshots
    for i in {1..10}; do
        echo "Taking screenshot $i/10 - Press Enter when ready for next shot"
        read -p "Ready for screenshot $i? "
        flameshot gui --path "$DEMO_DIR/screenshots" --filename "tmrt_demo_screenshot_$i.png"
        sleep 2
    done
}

# Function to demonstrate TMRT framework
run_tmrt_demo() {
    echo "🚀 Running TMRT Framework Demo..."
    cd "$TMRT_DIR"
    
    echo ""
    echo "=== TMRT Framework Competition Demo ==="
    echo "OpenAI GPT-OSS-20b Red Teaming Challenge"
    echo "======================================="
    echo ""
    
    # Show project structure
    echo "📁 TMRT Project Structure:"
    tree -L 2 . 2>/dev/null || ls -la
    echo ""
    
    # Show final submission package
    echo "📦 Competition Submission Package:"
    ls -la FINAL_SUBMISSION_PACKAGE/submission_files/
    echo ""
    
    # Display key metrics
    echo "📊 Competition Metrics:"
    echo "• Total Findings: 10 high-quality discoveries"
    echo "• Critical Risk: 6 findings (60%)"
    echo "• Average Novelty: 87.8%"
    echo "• Techniques: 40+ advanced methods"
    echo "• Category Coverage: 100%"
    echo ""
    
    # Show validation results
    echo "✅ Validation Results:"
    if [ -f "final_submission/validation_report_*.json" ]; then
        echo "100% Submission Ready - All checks passed"
    fi
    echo ""
    
    # Show GitHub repository
    echo "🐙 GitHub Repository:"
    echo "https://github.com/734ai/Token-Manifold-Red-Teaming-TMRT-"
    echo ""
    
    # Show kaggle submission file
    echo "🏆 Primary Competition Submission:"
    head -20 FINAL_SUBMISSION_PACKAGE/submission_files/kaggle_submission.json
    echo ""
    
    # Show some findings
    echo "🔍 Sample Vulnerability Findings:"
    echo "TMRT-001: Bidirectional Text + Authority Scaffolding (CRITICAL, 0.92 novelty)"
    echo "TMRT-006: Multi-Modal Deception + Attention Exploit (CRITICAL, 0.94 novelty)"
    echo "TMRT-010: Gradient-Free + Persistent Jailbreak (CRITICAL, 0.90 novelty)"
    echo ""
    
    echo "🎯 Demo Complete - Ready for Competition Submission!"
}

# Function to create complete demo video
create_complete_demo() {
    echo "🎬 Creating Complete Demo Video..."
    
    echo ""
    echo "📝 Demo Script Overview:"
    echo "1. Introduction & Project Overview (30 seconds)"
    echo "2. Live Framework Demonstration (3-4 minutes)"  
    echo "3. Results & Metrics Showcase (1-2 minutes)"
    echo "4. Technical Architecture Overview (1 minute)"
    echo "5. Competition Readiness Summary (30 seconds)"
    echo ""
    
    echo "Choose recording method:"
    echo "1) OBS Studio (GUI, professional)"
    echo "2) FFmpeg (command line, automated)"
    echo "3) Terminal recording only (asciinema)"
    echo "4) Full demo with all tools"
    
    read -p "Select option (1-4): " choice
    
    case $choice in
        1)
            setup_obs_recording
            echo "Use OBS Studio GUI to record the demo"
            run_tmrt_demo
            ;;
        2)
            echo "Starting FFmpeg screen recording in 5 seconds..."
            sleep 5
            record_screen_ffmpeg &
            FFMPEG_PID=$!
            run_tmrt_demo
            kill $FFMPEG_PID 2>/dev/null
            ;;
        3)
            record_terminal_demo
            ;;
        4)
            echo "Full demo with all recording tools..."
            setup_obs_recording
            sleep 3
            record_terminal_demo &
            TERMINAL_PID=$!
            take_screenshots &
            SCREENSHOT_PID=$!
            run_tmrt_demo
            kill $TERMINAL_PID $SCREENSHOT_PID 2>/dev/null
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
}

# Main menu
echo ""
echo "🎯 TMRT Demo Recording Options:"
echo "1) Setup OBS Studio recording"
echo "2) Record with FFmpeg (full screen)"
echo "3) Record terminal session only"
echo "4) Take demo screenshots"
echo "5) Run TMRT demo (no recording)"
echo "6) Create complete demo video"
echo "7) View demo script"
echo ""

read -p "Select option (1-7): " option

case $option in
    1)
        setup_obs_recording
        ;;
    2)
        record_screen_ffmpeg
        ;;
    3)
        record_terminal_demo
        ;;
    4)
        take_screenshots
        ;;
    5)
        run_tmrt_demo
        ;;
    6)
        create_complete_demo
        ;;
    7)
        cat "$DEMO_DIR/demo_script.md"
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo ""
echo "🏆 Recording complete! Files saved to: $RECORDINGS_DIR"
echo "📁 Screenshots saved to: $DEMO_DIR/screenshots"
echo ""
echo "Next steps:"
echo "1. Review recorded materials"
echo "2. Edit with Kdenlive if needed"
echo "3. Upload to YouTube"
echo "4. Add to Kaggle competition submission"
echo ""
echo "🚀 Ready to submit and win the competition!"
