# SNN-Fruit-Detection-and-Counting

# 🍎 Fruit Detection and Counting Using Spiking Neural Networks (SNN)

This project uses **Spiking Neural Networks (SNN)** to detect and count apples in video frames. It is implemented in a Jupyter notebook (`mini.ipynb`) and runs on Google Colab. The system processes each frame of the video, converts it into spike-based data, detects apples using an SNN model, and outputs a final video with bounding boxes and a count of detected apples.

---

## 🧠 Key Features

- 🧩 Spike-based image processing using **SpikingJelly**
- 🧠 Neuromorphic apple detection using **Spiking Neural Networks**
- 📹 Video frame extraction and apple detection using OpenCV
- 🧮 Duplicate removal for accurate apple counting
- 📊 Frame-wise and total apple count output
- 💾 Final video output with bounding boxes and labels

---

## 🚀 Tech Stack

- Python 3.x
- Google Colab
- SpikingJelly
- OpenCV
- NumPy
- Matplotlib
- PyTorch

---

## 📁 Folder Structure

.
├── mini.ipynb          # Main notebook
├── /frames             # (Optional) Video frames (auto-generated)
├── /outputs            # Output videos/images (auto-generated)
├── README.md           # Project documentation

---

## ⚙️ Setup & Installation

1. Clone the repository:

   git clone [https://github.com/your-username/SNN-Fruit-Detection-and-Counting.git
   cd SNN-Fruit-Detection-and-Counting](https://github.com/tulasipraneeth/SNN-Fruit-Detection-and-Counting.git)

2. Open `mini.ipynb` in **Google Colab**

3. Install the required dependencies inside the notebook:

   !pip install -q spikingjelly opencv-python-headless matplotlib

4. Upload your apple orchard video (.mp4) and run all cells.

---

## 📈 Sample Output

- Real-time apple detection in frames
- Bounding boxes drawn on each apple
- Total apple count displayed per frame and at the end

Example output snippet:

   Frame 1: 7 apples detected
   Frame 2: 6 apples detected
   ...
   Total apples: 58 (duplicates removed)

> (Optional) Add result images or videos in an `assets/` folder and update here.

---

## 📌 Dataset

Supported:
- Monastery Apple Dataset (MAD)
- Custom orchard videos (.mp4 format)

Replace the input video in the notebook with any orchard footage.

---

## ✅ Output

- Final video with apples outlined and counted
- Printed frame-wise and total apple count

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, improve the model, clean the code, or enhance the UI.

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- SpikingJelly for SNN frameworks
- MAD Dataset contributors
- Google Colab for compute resources

---

## 📬 Contact

**Mukka Tulasipraneeth**  
📧 mukkatulasipraneeth@example.com  
🔗 [https://www.linkedin.com/in/tulasipraneeth](https://www.linkedin.com/in/mukka-tulasipraneeth-34417028a/)

