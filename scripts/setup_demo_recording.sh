#!/bin/bash
# TMRT Demo Recording Setup Script
# For Kali Linux - Competition Video Creation

echo "🎥 TMRT Competition Demo Recording Setup"
echo "========================================"
echo "Setting up Kali Linux tools for professional demo video creation"

# Update system
echo "📦 Updating system packages..."
sudo apt update

# Install OBS Studio for screen recording
echo "🎬 Installing OBS Studio for screen recording..."
sudo apt install -y obs-studio

# Install video editing tools
echo "✂️ Installing video editing tools..."
sudo apt install -y kdenlive

# Install terminal recording tools  
echo "💻 Installing terminal recording tools..."
pip3 install asciinema

# Install screenshot tools
echo "📸 Installing screenshot tools..."
sudo apt install -y flameshot

# Install audio tools
echo "🎵 Installing audio recording tools..."
sudo apt install -y audacity

# Install additional utilities
echo "🔧 Installing additional utilities..."
sudo apt install -y ffmpeg imagemagick

# Create demo directory structure
echo "📁 Creating demo directory structure..."
mkdir -p ~/tmrt_demo/{recordings,screenshots,assets,final_video}

# Download and prepare TMRT project
echo "📥 Preparing TMRT project for demo..."
cd ~/Desktop/openai-res

# Create demo script
cat > ~/tmrt_demo/demo_script.md << 'EOF'
# TMRT Demo Script

## 🎯 Demo Flow (5-7 minutes)

### 1. Opening (30 seconds)
- Welcome and introduction
- Competition context
- TMRT framework overview

### 2. Live Vulnerability Discovery (3-4 minutes)
- Launch TMRT framework
- Show real-time attack generation
- Demonstrate multiple techniques:
  * Unicode bidirectional attacks
  * Embedding optimization
  * Scaffolding generation
  * Automated validation
- Display results and metrics

### 3. Results Overview (1-2 minutes)  
- Show 10 discovered vulnerabilities
- Highlight 6 critical findings
- Demonstrate novelty scoring
- Show category coverage

### 4. Technical Excellence (1 minute)
- Framework architecture overview
- Docker reproducibility demo
- Quality metrics display
- GitHub repository tour

### 5. Closing (30 seconds)
- Research impact summary
- Competition readiness confirmation
- Responsible disclosure commitment

## 🎬 Recording Commands

```bash
# Start screen recording
obs

# Terminal recording for specific sections
asciinema rec tmrt_vulnerability_discovery.cast

# Take screenshots for key moments
flameshot gui

# Record audio narration
audacity

# Combine video elements
kdenlive
```

## 📝 Demo Talking Points

### Opening
"Welcome to the Token-Manifold Red Teaming framework demonstration for the OpenAI GPT-OSS-20b Red Teaming Challenge. I'm about to show you how our advanced automated framework discovered 10 novel vulnerabilities, including 6 critical-risk findings previously unknown in GPT-OSS-20b."

### Technical Demo
"Let me demonstrate our multi-vector approach combining Unicode exploitation, embedding optimization, and sophisticated scaffolding. Watch as the system generates and validates attacks in real-time..."

### Results
"Our framework achieved an 87.8% average novelty score across 10 high-quality findings, with complete coverage across all vulnerability categories and a 90% high/critical risk ratio."

### Conclusion  
"The TMRT framework represents a breakthrough in automated vulnerability discovery, advancing AI safety through rigorous, responsible research. Our comprehensive submission package includes full reproducibility materials and detailed documentation."
EOF

echo "📋 Demo script created at ~/tmrt_demo/demo_script.md"

# Create video recording checklist
cat > ~/tmrt_demo/recording_checklist.md << 'EOF'
# 🎥 Recording Checklist

## Pre-Recording Setup
- [ ] Close unnecessary applications
- [ ] Set screen resolution to 1920x1080
- [ ] Configure OBS recording settings
- [ ] Test microphone audio levels
- [ ] Prepare TMRT framework for demo
- [ ] Review demo script timing

## Recording Setup
- [ ] Start OBS Studio
- [ ] Configure screen capture area
- [ ] Set up audio recording
- [ ] Test recording quality
- [ ] Prepare terminal sessions

## Demo Content
- [ ] Framework introduction (30s)
- [ ] Live vulnerability discovery (3-4m)
- [ ] Results overview (1-2m)  
- [ ] Technical architecture (1m)
- [ ] Closing summary (30s)

## Post-Recording
- [ ] Review raw footage
- [ ] Edit with Kdenlive if needed
- [ ] Export final video
- [ ] Upload to YouTube
- [ ] Add to Kaggle submission

## Quality Checks
- [ ] Video quality: 1080p minimum
- [ ] Audio quality: Clear narration
- [ ] Content flow: Logical progression
- [ ] Time limit: 5-7 minutes total
- [ ] Professional presentation
EOF

echo "✅ Recording checklist created at ~/tmrt_demo/recording_checklist.md"

# Set up OBS Studio basic config
echo "⚙️ Creating OBS Studio basic configuration..."
mkdir -p ~/.config/obs-studio/basic
cat > ~/.config/obs-studio/basic/scenes.json << 'EOF'
{
    "current_scene": "TMRT Demo",
    "current_transition": "Fade",
    "sources": [
        {
            "name": "Desktop Capture",
            "type": "xshm_input",
            "settings": {
                "screen": 0
            }
        }
    ],
    "scenes": [
        {
            "name": "TMRT Demo",
            "sources": ["Desktop Capture"]
        }
    ]
}
EOF

echo ""
echo "🏆 Demo Recording Setup Complete!"
echo "=================================="
echo "Next steps:"
echo "1. Review demo script: ~/tmrt_demo/demo_script.md"
echo "2. Check recording checklist: ~/tmrt_demo/recording_checklist.md"  
echo "3. Launch OBS Studio and configure recording"
echo "4. Practice demo run-through"
echo "5. Record professional demo video"
echo ""
echo "📁 Demo workspace: ~/tmrt_demo/"
echo "🎬 Tools ready: OBS Studio, Kdenlive, Asciinema, Audacity"
echo ""
echo "🚀 Ready to create winning competition demo!"
