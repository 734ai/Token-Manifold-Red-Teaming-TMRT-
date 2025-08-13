# 🎬 TMRT Competition Demo Recording Guide

## 🎯 **Demo Structure (5-7 minutes total)**

### **1. Introduction (30 seconds)**
**What to say:**
"Welcome to the Token-Manifold Red Teaming framework demonstration for the OpenAI GPT-OSS-20b Red Teaming Challenge. I'm about to show you how our advanced automated framework discovered 10 novel vulnerabilities with an 87.8% average novelty score, including 6 critical-risk findings."

**What to show:**
- Open terminal in `/home/o1/Desktop/openai-res/`
- Display project structure: `ls -la`

### **2. Live Demonstration (4 minutes)**
**What to say:**
"Let me demonstrate our multi-vector approach combining Unicode exploitation, embedding optimization, and sophisticated scaffolding."

**Commands to run:**
```bash
# Show framework components
ls src/tmrt/
echo "6 core modules implementing 40+ advanced techniques"

# Show key findings
cat FINAL_SUBMISSION_PACKAGE/submission_files/kaggle_submission.json | head -20
echo "10 high-quality findings discovered"

# Display findings summary
python -c "
import json
with open('FINAL_SUBMISSION_PACKAGE/submission_files/kaggle_submission.json') as f:
    data = json.load(f)
print('🎯 Competition Findings Summary:')
print(f'Total Findings: {len(data[\"competition_entries\"])}')
critical = sum(1 for e in data['competition_entries'] if e.get('risk_assessment') == 'CRITICAL')
high = sum(1 for e in data['competition_entries'] if e.get('risk_assessment') == 'HIGH')
print(f'Critical Risk: {critical}')
print(f'High Risk: {high}')
avg_novelty = sum(e.get('novelty_score', 0) for e in data['competition_entries']) / len(data['competition_entries'])
print(f'Average Novelty: {avg_novelty:.1%}')
"

# Show validation results
echo "100% submission validation:"
python scripts/final_submission_validator.py | tail -5
```

**What to show in editor:**
- Open `COMPETITION_WRITEUP.md` and scroll through key sections
- Show the findings table with critical vulnerabilities
- Display GitHub repository

### **3. Results Showcase (2 minutes)**
**What to say:**
"Our framework achieved exceptional results across all evaluation criteria."

**Commands to run:**
```bash
# Show submission package
ls -la FINAL_SUBMISSION_PACKAGE/
echo "Complete submission package ready"

# Display key metrics
echo "🏆 Key Achievements:"
echo "• 10 High-quality findings"
echo "• 87.8% average novelty score" 
echo "• 6 critical + 3 high-risk vulnerabilities"
echo "• 40+ advanced techniques"
echo "• 100% category coverage"
echo "• Complete reproducibility package"
```

**What to show in editor:**
- Open findings table in `COMPETITION_WRITEUP.md`
- Highlight the 6 critical vulnerabilities
- Show research contributions section

### **4. Technical Excellence (1 minute)**
**What to say:**
"The framework includes comprehensive reproducibility materials and documentation."

**Commands to run:**
```bash
# Show Docker setup
cat docker/Dockerfile | head -10
echo "Complete Docker environment for reproducibility"

# Show documentation
ls technical_documentation/
ls FINAL_SUBMISSION_PACKAGE/
echo "35+ pages of comprehensive documentation"

# Show GitHub repository
echo "🔗 GitHub: https://github.com/734ai/Token-Manifold-Red-Teaming-TMRT-"
```

### **5. Conclusion (30 seconds)**
**What to say:**
"The TMRT framework represents breakthrough research in automated vulnerability discovery. Our submission is ready for the competition with comprehensive documentation, full reproducibility, and significant research contributions to AI safety. We discovered 6 critical vulnerabilities in GPT-OSS-20b that advance the state-of-the-art in AI red-teaming."

**Final display:**
- Show the competition writeup title
- Display "100% SUBMISSION READY" status

## 🎬 **Recording Commands**

### **Quick 7-minute recording (RECOMMENDED):**
```bash
cd /home/o1/tmrt_demo_recording
./quick_record.sh
```

### **Professional segmented recording:**
```bash
cd /home/o1/tmrt_demo_recording  
./record_demo_sequence.sh
```

## 🔧 **Pre-Recording Checklist**
- [ ] Terminal open in `/home/o1/Desktop/openai-res/`
- [ ] Editor open with `COMPETITION_WRITEUP.md`
- [ ] Clear desktop (close unnecessary windows)
- [ ] Test microphone audio
- [ ] Review talking points
- [ ] Practice command sequence

## 🎯 **Recording Tips**
- Speak clearly and at moderate pace
- Allow brief pauses between sections
- Emphasize key numbers (87.8% novelty, 6 critical findings)
- Show confidence and expertise
- Keep energy high throughout

## 📊 **Key Numbers to Emphasize**
- **10 findings** discovered
- **87.8% average novelty** score
- **6 critical vulnerabilities**
- **40+ advanced techniques**
- **100% validation score**
- **100% category coverage**

Ready to create your winning demo! 🏆
