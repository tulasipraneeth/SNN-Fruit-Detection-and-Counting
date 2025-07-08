# 🍎 SNN-Fruit-Detection-and-Counting

A futuristic AI-powered system for detecting and counting apples using **Spiking Neural Networks (SNN)**.  
It supports both:

- 🧠 `mini.ipynb`: Neuromorphic detection in Google Colab (Jupyter Notebook)
- 🌐 `app.py`: Interactive Web App with Streamlit and futuristic UI

---

## 🧠 Key Features

- 🧩 Spike-based image encoding using SpikingJelly & snnTorch
- 📹 Video frame extraction & apple detection via OpenCV
- 🔁 Duplicate removal for accurate counting
- 🧮 Frame-wise and total apple count display
- 💾 Final video output with bounding boxes & labels
- 🎨 Futuristic 3D animated UI with light/dark mode
- ⏱️ Real-time processing progress bar

---

## 🚀 Tech Stack

| Component | Description |
|----------|-------------|
| 🧠 Model | Spiking Neural Network (Leaky Integrate-and-Fire) |
| 📊 Frameworks | PyTorch, snnTorch, OpenCV, NumPy, SciPy |
| 📘 Notebook | Google Colab (`mini.ipynb`) |
| 🌐 Web App | Streamlit (`app.py`) with advanced UI |

---

## 📁 Folder Structure

.
├── mini.ipynb          # Jupyter Notebook for detection in Colab
├── app.py              # Streamlit-based web UI
├── requirements.txt    # Dependencies for app.py
├── README.md           # Project documentation
├── /outputs            # Output videos (auto-generated)

---

## ⚙️ Setup & Installation

### 🔬 For Notebook (`mini.ipynb`)

1. Clone the repo or open on Colab:

   ```bash
   git clone https://github.com/tulasipraneeth/SNN-Fruit-Detection-and-Counting.git
   cd SNN-Fruit-Detection-and-Counting
   ```

2. Open `mini.ipynb` in **Google Colab**

3. Install dependencies inside the notebook:

   ```python
   !pip install -q spikingjelly opencv-python-headless matplotlib
   ```

4. Upload your `.mp4` apple orchard video and run all cells.

---

### 🌐 For Web App (`app.py`)

1. Clone the repository:

   ```bash
   git clone https://github.com/tulasipraneeth/SNN-Fruit-Detection-and-Counting.git
   cd SNN-Fruit-Detection-and-Counting
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Open browser at `http://localhost:8501`

---

## 🖼️ UI Preview

> Add screenshots or videos in `/assets` folder  
> Example:

![demo](assets/ai-snn-ui-demo.gif)

---

## 📈 Sample Output

Frame 1: 7 apples detected  
Frame 2: 6 apples detected  
...  
Total apples: 58 (duplicates removed)

Output video includes:
- 🟩 Bounding boxes
- 🍏 Confidence scores
- 🧮 Live frame count

---

## 🧪 Dataset Support

Supports:
- ✅ Monastery Apple Dataset (MAD)
- ✅ Any custom `.mp4` orchard videos

Just upload/replace your video to begin.

---

## 🧩 Customization Options

| Feature           | Supported |
|------------------|-----------|
| Frame resolution | ✅ Scalable |
| Model threshold  | ✅ Adjustable |
| UI theme toggle  | ✅ Light/Dark |
| Live chart       | ✅ Frame-wise apple count |
| Auto de-duplication | ✅ Yes |

---

## 🧠 Future Enhancements

- 🔁 Live camera feed (real-time orchard detection)
- 💬 Audio alerts for every detection
- 📊 Apple size or ripeness estimation

---

## 🤝 Contributing

Contributions welcome!  
You can:
- Improve detection logic
- Enhance UI animations
- Add new datasets
- Optimize spike encoding

To contribute:

```bash
fork → edit → pull request ✅
```

---

## 📜 License

Licensed under the **MIT License**  
Free to use, distribute, and modify with attribution.

---

## 🙏 Acknowledgements

- 🧠 SpikingJelly
- 🍎 MAD Apple Dataset
- 🚀 Streamlit
- 💻 Google Colab

---

## 👤 Author & Contact

**Mukka Tulasipraneeth**  
📧 mukkatulasipraneeth@example.com  
🔗 [LinkedIn](https://www.linkedin.com/in/mukka-tulasipraneeth-34417028a/)  
🐙 [GitHub](https://github.com/tulasipraneeth/SNN-Fruit-Detection-and-Counting)

---

> ⭐ If you like this project, give it a star on GitHub!
