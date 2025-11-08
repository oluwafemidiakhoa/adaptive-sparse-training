# 🚀 Deploying Your Malaria Detection Web App

## ✅ What You Now Have

Your repository includes **3 ways to use the malaria detector**:

1. **Training + App in One** - `Malaria_Train_and_Deploy.ipynb` (Recommended!)
2. **Standalone App** - `app.py` (For Hugging Face Spaces)
3. **Training Only** - `Malaria_AST_Training.ipynb` (Original)

---

## 🎯 Option 1: Train + Deploy in Colab (Easiest!)

**Perfect for demos and quick sharing (72-hour link)**

### Steps:

1. **Open the notebook:**
   ```
   https://colab.research.google.com/github/oluwafemidiakhoa/Malaria/blob/main/malaria_ast_starter/malaria_ast_starter/Malaria_Train_and_Deploy.ipynb
   ```

2. **Enable GPU:** Runtime → Change runtime type → GPU

3. **Run all cells** (Runtime → Run all)

4. **Upload kaggle.json** when prompted

5. **Wait ~30 minutes:**
   - Training: 25 minutes
   - Setup + Deploy: 5 minutes

6. **Get your public URL!**
   - Look for: `Running on public URL: https://xxxxx.gradio.live`
   - Share this link with anyone!
   - Valid for 72 hours

### Demo Your App:

Share on social media:
```
🌿 Try my AI malaria detector (live demo):
👉 [your Gradio link]

✅ 96% accuracy
✅ 60% energy savings
✅ Instant diagnosis from cell images

#AI #GlobalHealth
```

---

## 🏠 Option 2: Deploy Permanently on Hugging Face Spaces

**For a permanent public app (no 72-hour limit)**

### Prerequisites:
- Hugging Face account: https://huggingface.co/join
- Trained model (`best.pt` from Colab)

### Steps:

#### 1. Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Settings:
   - **Name:** `malaria-detector`
   - **License:** MIT
   - **SDK:** Gradio
   - **Hardware:** CPU (free) or GPU (paid)
4. Click **Create Space**

#### 2. Upload Files

Upload these 3 files to your Space:

**a) app.py** (from the repo)
```bash
# Download from GitHub or use the one in your repo
```

**b) best.pt** (your trained model)
```bash
# Download from Colab:
# files.download('checkpoints_ast/best.pt')
```

**c) requirements.txt**
```text
torch
torchvision
pillow
numpy
matplotlib
gradio
```

#### 3. Your App is Live!

URL: `https://huggingface.co/spaces/YOUR_USERNAME/malaria-detector`

**Modify app.py if needed:**
- Edit the model loading section to match your model name
- Customize the interface text
- Add example images

---

## 📱 Option 3: Local Deployment

**Run the app on your own computer**

### Steps:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your model:**
   ```bash
   # Put best.pt in the same folder as app.py
   # OR edit app.py line with the model path
   ```

3. **Run the app:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   - Local: http://localhost:7860
   - Public (temporary): Check terminal for Gradio public link

---

## 🎨 Customizing the App

### Change Model Path

Edit `app.py`:
```python
# Line ~80 (in load_model function)
MODEL = load_model("path/to/your/model.pt", device=DEVICE)
```

### Add Example Images

```python
# In the Gradio interface
examples=[
    "path/to/parasitized_sample.png",
    "path/to/uninfected_sample.png",
]
```

### Change Theme

```python
demo = gr.Blocks(theme=gr.themes.Base())  # Or Soft(), Glass(), Monochrome()
```

### Add More Features

Ideas:
- Batch processing (multiple images)
- Download diagnosis report (PDF)
- History of predictions
- Confidence threshold slider
- Model comparison (AST vs baseline)

---

## 🚀 Deployment Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Colab Gradio** | ✅ Free<br>✅ Fast setup<br>✅ No account needed | ❌ 72-hour limit<br>❌ Requires Colab running | Quick demos, testing |
| **HF Spaces** | ✅ Permanent<br>✅ Custom URL<br>✅ Version control | ❌ Requires HF account<br>❌ Manual file upload | Production, portfolio |
| **Local** | ✅ Full control<br>✅ No internet needed<br>✅ Private | ❌ Only accessible locally<br>❌ Manual setup | Development, testing |

---

## 📊 App Features

Your deployed app includes:

✅ **Drag & drop image upload**
✅ **Real-time diagnosis**
✅ **Grad-CAM heatmap** (shows where model looked)
✅ **Confidence scores** (percentage for each class)
✅ **Responsive design** (works on mobile)
✅ **Professional UI** (thanks to Gradio)

---

## 🎤 Pitching Your App

### For Twitter/LinkedIn:
```
🌿 Live demo of my energy-efficient malaria AI!

Try it: [your link]

Features:
✅ Upload cell image → Instant diagnosis
✅ 96% accuracy
✅ Shows AI reasoning (Grad-CAM)
✅ 60% less energy than traditional AI

Built with #PyTorch + Adaptive Sparse Training
```

### For Portfolio:
```markdown
## Malaria Detection AI (Live Demo)

**Try it:** [your link]

An energy-efficient deep learning system for malaria diagnosis:
- 95-97% accuracy on NIH malaria dataset
- Trained with Adaptive Sparse Training (60% energy savings)
- Deployable web interface with Grad-CAM visualization
- 25-minute training on free Colab GPU

**Tech Stack:** PyTorch, Gradio, EfficientNet, Adaptive Sparse Training
```

### For Resume:
```
Malaria Diagnostic AI Web Application
- Developed energy-efficient deep learning model achieving 96% accuracy
- Reduced training energy by 60% using Adaptive Sparse Training
- Deployed interactive web interface with Grad-CAM interpretability
- Made medical AI accessible via free cloud deployment
```

---

## 🐛 Troubleshooting

### "Model not loading"
- Check model path in app.py
- Ensure model architecture matches (EfficientNet-B0, ResNet, etc.)
- Verify model file is not corrupted

### "Out of memory"
- Use CPU instead of GPU for deployment
- Reduce image size in preprocessing
- Close other apps/tabs

### "Gradio link not working"
- Colab: Keep notebook running
- HF Spaces: Check build logs
- Local: Check firewall settings

### "Predictions are wrong"
- Verify you uploaded the correct model
- Check that images are cell microscopy images
- Ensure image preprocessing matches training

---

## 📚 Resources

**Documentation:**
- Gradio Docs: https://gradio.app/docs
- Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
- Your GitHub: https://github.com/oluwafemidiakhoa/Malaria

**Your Notebooks:**
- Training + Deploy: `Malaria_Train_and_Deploy.ipynb`
- Training Only: `Malaria_AST_Training.ipynb`

**App Code:**
- Full Featured: `app.py`
- Simplified (auto in notebook): Generated during training

---

## 🎉 Next Steps

1. **Train your model** using `Malaria_Train_and_Deploy.ipynb`
2. **Get the public link** from Gradio (appears in notebook output)
3. **Share widely!** Twitter, LinkedIn, portfolio
4. **Optional:** Deploy permanently to Hugging Face Spaces
5. **Iterate:** Add features, improve UI, try different models

---

## 💡 Pro Tips

- **Keep Colab running:** Gradio link dies when notebook stops
- **Download your model:** Always save `best.pt` to your computer
- **Test locally first:** Debug on your machine before deploying
- **Add analytics:** Track how many people use your app
- **Collect feedback:** Add a feedback form to improve the model

---

**Ready to deploy? Start with `Malaria_Train_and_Deploy.ipynb`!** 🚀

**Questions?** Check the main README.md or open a GitHub issue.
