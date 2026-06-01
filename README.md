# Pushkal Gupta

**India** &nbsp;·&nbsp; **Edge AI & on-device intelligence** &nbsp;·&nbsp; **Firmware × ML**

**Languages**
&nbsp;
[![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)](#) [![C](https://img.shields.io/badge/-C-00599C?style=flat-square&logo=c&logoColor=white)](#) [![C++](https://img.shields.io/badge/-C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)](#) [![Julia](https://img.shields.io/badge/-Julia-9558B2?style=flat-square&logo=julia&logoColor=white)](#) [![Java](https://img.shields.io/badge/-Java-007396?style=flat-square&logo=openjdk&logoColor=white)](#) [![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](#) [![R](https://img.shields.io/badge/-R-276DC3?style=flat-square&logo=r&logoColor=white)](#) [![PostgreSQL](https://img.shields.io/badge/-PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)](#) [![MySQL](https://img.shields.io/badge/-MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)](#) [![HTML5](https://img.shields.io/badge/-HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)](#) [![CSS3](https://img.shields.io/badge/-CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)](#)

**ML &amp; Data**
&nbsp;
[![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#) [![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](#) [![Keras](https://img.shields.io/badge/-Keras-D00000?style=flat-square&logo=keras&logoColor=white)](#) [![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](#) [![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](#) [![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](#) [![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](#) [![Ollama](https://img.shields.io/badge/-Ollama-000000?style=flat-square&logo=ollama&logoColor=white)](#) [![Selenium](https://img.shields.io/badge/-Selenium-43B02A?style=flat-square&logo=selenium&logoColor=white)](#) [![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)

**Embedded &amp; Edge**
&nbsp;
[![Zephyr RTOS](https://img.shields.io/badge/-Zephyr_RTOS-7929D2?style=flat-square&logo=zephyrproject&logoColor=white)](#) [![FreeRTOS](https://img.shields.io/badge/-FreeRTOS-008000?style=flat-square&logo=freertos&logoColor=white)](#) [![Nordic nRF](https://img.shields.io/badge/-Nordic_nRF-00A9CE?style=flat-square&logoColor=white)](#) [![Memfault](https://img.shields.io/badge/-Memfault-5C45FF?style=flat-square&logoColor=white)](#) [![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](#)

**Tools**
&nbsp;
[![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white)](#) [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](#) [![Linux](https://img.shields.io/badge/-Linux-FCC624?style=flat-square&logo=linux&logoColor=black)](#) [![Anaconda](https://img.shields.io/badge/-Anaconda-44A833?style=flat-square&logo=anaconda&logoColor=white)](#) [![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](#) [![AWS](https://img.shields.io/badge/-AWS-232F3E?style=flat-square&logo=amazonwebservices&logoColor=white)](#) [![Tableau](https://img.shields.io/badge/-Tableau-E97627?style=flat-square&logo=tableau&logoColor=white)](#) [![VS Code](https://img.shields.io/badge/-VS_Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white)](#) [![Excel](https://img.shields.io/badge/-Excel-217346?style=flat-square&logo=microsoftexcel&logoColor=white)](#)

> I build machine learning that runs on real hardware — from the autograd math up to the sensor on the device. B.Tech CS (IoT) @ VIT Vellore. I like to write the naive version from scratch, prove it with tests, look at the output, then reach for the library.

## What I'm doing

- **Understanding by building** — autograd engines, neural nets, and classical vision operators written from first principles. If I can't derive it, I don't trust the abstraction.
- **On-device ML** — the path that actually matters: train, compress, deploy, benchmark on hardware. Latency and memory measured on real silicon, not just on a laptop.
- **Sensor-flavoured systems** — noisy time-series, industrial sensor data, edge–fog–cloud architectures where a millisecond decision happens locally.

## Built from scratch

The thread that ties most of my work together: write it from the math up, test every piece, then look at what it actually does.

- **[micrograd](https://github.com/Pushkal-Gupta/micrograd)** — A scalar-level autograd engine. Each node holds one value and the op that made it; the backward pass walks the DAG in reverse applying the chain rule. Built across basics → intermediate → advanced notebooks, with the gradient-accumulation failure mode for reused nodes made explicit.
- **[NumPy-only Neural Net](https://github.com/Pushkal-Gupta/Numpy-only-Neural-Net)** — A feedforward net in pure NumPy, no autograd framework. Three notebooks in sequence: a heavily-annotated MNIST digit classifier (every matrix shape and gradient derived by hand), a deeper Fashion-MNIST net with mini-batch SGD, and a dynamic-depth MLP for letter recognition with He init and per-epoch shuffling. Picks up where micrograd leaves off: scalar autodiff → vectorized matrices.
- **[ImageLab](https://github.com/Pushkal-Gupta/Julia)** *(Julia)* — Classical computer vision, built operator by operator: naive convolution → six padding modes → separable filters → first/second-order edges → Canny from scratch → Harris corners, Hough lines, connected components, NCC → Gaussian/Laplacian pyramids → anisotropic diffusion → multi-band blending → convolution as Toeplitz/circulant matrices → a Whitted-style ray tracer → differentiable filters via ForwardDiff → Lucas-Kanade and Horn-Schunck optical flow (both pyramidal) → colour spaces and Di Zenzo gradients. **1530 passing tests.**

## Applied ML & data

- **[Semiconductor Yield Prediction & Defect Analysis](https://github.com/Pushkal-Gupta/Semiconductor-Data-Analysis)** — Two questions on real fab data. *Will a wafer fail final test?* — UCI SECOM, 590 in-line sensor signals reduced to 271 clean features, then LogReg / gradient boosting / MLP compared with SHAP, SMOTE, and threshold tuning. *What defect pattern is on the wafer map?* — Kaggle WM-811K, a small class-weighted CNN that nails the visually distinct defects (Edge-Ring F1 0.97) and struggles on the same ambiguous ones (Loc, Scratch) a human would.
- **[Vehicle Telematics Analysis](https://github.com/Pushkal-Gupta/Hackathon/tree/main/MotorQ_DS)** *(MotorQ DS Hackathon 2025)* — An ingestion pipeline unifying four heterogeneous telematics sources (telemetry, triggers, mapping, synthetic) with 78–87% missing fields. Timestamps normalized across IST/UTC clock drift, odometer sensor-fault outliers (15% of readings) stripped via IQR, ignition events extracted by state-transition logic with source-priority conflict resolution (SYN > TRG > TLM), and charging sessions detected with Savitzky-Golay smoothing and battery-delta thresholds.
- **[AI Web Scraper](https://github.com/Pushkal-Gupta/Web-Scraper-AI)** — Prompt-driven extraction instead of fragile selectors. Selenium + BeautifulSoup pull page context; a locally-hosted Llama 3.1 (via Ollama, orchestrated with LangChain) interprets it into clean structured tables from a natural-language query.

## Edge, IoT & systems

- **[Fog-Based Vehicle Monitoring](https://github.com/Pushkal-Gupta/Fog-Based-Vehicle-Monitoring)** — A three-tier edge–fog–cloud architecture for real-time vehicle health. Millisecond safety decisions run locally; the cloud is reserved for non-critical analytics. Compact health-vectors (RMS, FFT spectral features, rate-of-change) cut telemetry volume while preserving signal for fleet-level RUL estimation.
- **[Graph-Based Location Privacy for IoT](https://github.com/Pushkal-Gupta/Graph-based_Location-Privacy-IoT)** — An object-oriented framework comparing five privacy-preserving algorithms on Microsoft's GeoLife GPS trajectory dataset, quantifying privacy–utility trade-offs on real-world IoT location streams.

## Web

- **[PG Hub](https://pushkalgupta.com)** — [[repo](https://github.com/Pushkal-Gupta/WebDev)] My personal web platform (React / Vite / Supabase, with auth), a growing home for things I build. It currently hosts **PGcode**, a NeetCode-style DSA platform with a Monaco editor, Judge0 execution, and a 22-topic DAG roadmap, and **PG.Chess**, multiplayer chess with Stockfish 18 WASM, Glicko-2 ratings, and real-time sync.

## Experience

**Ultrahuman** — Firmware & AI/ML Developer

- Designed a step-counting algorithm on accelerometer data hitting ~95% accuracy, and built a TensorFlow Lite walk vs. non-walk classifier to cut false positives in low-movement contexts.
- Integrated the algorithms into Zephyr RTOS firmware (12 Hz / 100 Hz sampling), resolved zero-step bugs, and led Base-1 testing to ≤10% step deviation against an Apple Watch Ultra.
- Shipped to production wearables via secure OTA updates, working across Science and Firmware teams.

## Writing

I write about the things that break when you build from scratch — autograd internals, how data flows to edge devices, and the latency-vs-accuracy reality of on-device inference.

[![Medium](https://img.shields.io/badge/-Medium-000000?style=flat-square&logo=medium&logoColor=white)](https://medium.com/@pushkalgupta) [![Substack](https://img.shields.io/badge/-Substack-FF6719?style=flat-square&logo=substack&logoColor=white)](https://substack.com/@pushkalgupta)

## Recognition

University Topper (Rank 1) all three years at VIT Vellore, CGPA 9.91 / 10.

## Activity

<!-- Snake animation: needs the GitHub Action in .github/workflows/snake.yml (provided). Run it once from the Actions tab to create the "output" branch. -->
[![Snake](https://raw.githubusercontent.com/Pushkal-Gupta/Pushkal-Gupta/output/snake.svg)](https://github.com/Pushkal-Gupta)

[![Stats](https://github-readme-stats.vercel.app/api?username=Pushkal-Gupta&show_icons=true&include_all_commits=true&count_private=true&hide_border=true&theme=tokyonight)](https://github.com/Pushkal-Gupta) [![Streak](https://github-readme-streak-stats.herokuapp.com/?user=Pushkal-Gupta&hide_border=true&theme=tokyonight)](https://github.com/Pushkal-Gupta)

## Connect

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pushkal-gupta) [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Pushkal-Gupta) [![LeetCode](https://img.shields.io/badge/-LeetCode-FFA116?style=flat-square&logo=leetcode&logoColor=black)](https://leetcode.com/u/Pushkal-Gupta/) [![Medium](https://img.shields.io/badge/-Medium-000000?style=flat-square&logo=medium&logoColor=white)](https://medium.com/@pushkalgupta) [![Substack](https://img.shields.io/badge/-Substack-FF6719?style=flat-square&logo=substack&logoColor=white)](https://substack.com/@pushkalgupta) [![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:pushkalgupta2005@gmail.com) [![Website](https://img.shields.io/badge/-Website-4285F4?style=flat-square&logo=googlechrome&logoColor=white)](https://pushkalgupta.com)

---

*Understanding systems from first principles. Building to understand.*

*Carpe Noctem.*
