"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// --- Type คร่าว ๆ ของ OpenCV ---
type CvType = any;

export default function Home() {
  // =====================================================
  // Refs: Video / Canvas / Camera
  // =====================================================
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const runningRef = useRef<boolean>(false);
  const lastUpdateRef = useRef<number>(0);

  // =====================================================
  // Refs: AI / Model
  // =====================================================
  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // =====================================================
  // State: UI
  // =====================================================
  const [status, setStatus] = useState("กำลังโหลด...");
  const [emotion, setEmotion] = useState("-");
  const [confidence, setConf] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);

  // =====================================================
  // Emotion smoothing (Method 1: Majority Vote)
  // =====================================================
  const emotionHistoryRef = useRef<Record<number, string[]>>({});
  const SMOOTH_BUFFER_SIZE = 5;

  


  function smoothEmotion(faceId: number, newLabel: string) {
    if (!emotionHistoryRef.current[faceId]) {
      emotionHistoryRef.current[faceId] = [];
    }

    const buf = emotionHistoryRef.current[faceId];
    buf.push(newLabel);
    if (buf.length > SMOOTH_BUFFER_SIZE) buf.shift();

    const count: Record<string, number> = {};
    for (const l of buf) count[l] = (count[l] || 0) + 1;

    let best = newLabel;
    let bestCount = 0;
    for (const [k, v] of Object.entries(count)) {
      if (v > bestCount) {
        best = k;
        bestCount = v;
      }
    }
    return best;
  }

  // =====================================================
  // Load OpenCV
  // =====================================================
  async function loadOpenCV() {
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        const wait = () => {
          if (cv?.Mat) {
            cvRef.current = cv;
            resolve();
          } else setTimeout(wait, 50);
        };
        cv.onRuntimeInitialized ? (cv.onRuntimeInitialized = wait) : wait();
      };
      script.onerror = () => reject(new Error("โหลด OpenCV ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // =====================================================
  // Load Haar Cascade
  // =====================================================
  async function loadCascade() {
    const cv = cvRef.current;
    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    const data = new Uint8Array(await res.arrayBuffer());

    const path = "face.xml";
    try {
      cv.FS_unlink(path);
    } catch {}
    cv.FS_createDataFile("/", path, data, true, false, false);

    const classifier = new cv.CascadeClassifier();
    classifier.load(path);
    faceCascadeRef.current = classifier;
  }

  // =====================================================
  // Load ONNX Model
  // =====================================================
  async function loadModel() {
    sessionRef.current = await ort.InferenceSession.create(
      "./models/emotion_yolo.onnx",
      { executionProviders: ["wasm"] }
    );

    const res = await fetch("/models/classes.json");
    classesRef.current = await res.json();
  }

  // =====================================================
  // Preprocess + Softmax
  // =====================================================
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;

    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const img = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        float[idx++] = img[i * 4 + c] / 255;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // =====================================================
  // Main Loop (Multi-face + Smoothing)
  // =====================================================
  async function loop() {
    if (!runningRef.current || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    if (video.videoWidth === 0) {
      requestAnimationFrame(loop);
      return;
    }

    let src: any = null;
    let gray: any = null;
    let faces: any = null;

    try {
      const cv = cvRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.drawImage(video, 0, 0);

      src = cv.imread(canvas);
      gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      faces = new cv.RectVector();
      faceCascadeRef.current.detectMultiScale(gray, faces, 1.1, 3);

      const maxFaces = 3;
      const faceCount = Math.min(faces.size(), maxFaces);

      for (let i = 0; i < faceCount; i++) {
        const r = faces.get(i);

        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(r.x, r.y, r.width, r.height);

        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = r.width;
        faceCanvas.height = r.height;

        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(
          canvas,
          r.x,
          r.y,
          r.width,
          r.height,
          0,
          0,
          r.width,
          r.height
        );

        const input = preprocessToTensor(faceCanvas);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[sessionRef.current!.inputNames[0]] = input;

        const out = await sessionRef.current!.run(feeds);
        const logits =
          out[sessionRef.current!.outputNames[0]].data as Float32Array;

        const probs = softmax(logits);
        let maxIdx = 0;
        for (let j = 1; j < probs.length; j++) {
          if (probs[j] > probs[maxIdx]) maxIdx = j;
        }

        const rawLabel = classesRef.current![maxIdx];
        const confVal = probs[maxIdx];
        const smoothLabel = smoothEmotion(i, rawLabel);

        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(r.x, Math.max(0, r.y - 26), 180, 26);
        ctx.fillStyle = "white";
        ctx.font = "14px sans-serif";
        ctx.fillText(
          `${smoothLabel} ${(confVal * 100).toFixed(0)}%`,
          r.x + 6,
          r.y - 8
        );

        // UI ด้านบน: ใช้คนแรก
        if (i === 0) {
          const now = Date.now();
          if (now - lastUpdateRef.current > 200) {
            setEmotion(smoothLabel);
            setConf(confVal);
            lastUpdateRef.current = now;
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      if (src) src.delete();
      if (gray) gray.delete();
      if (faces) faces.delete();
    }

    if (runningRef.current) requestAnimationFrame(loop);
  }

  // =====================================================
  // Start / Stop Camera
  // =====================================================
  async function startCamera() {
    if (runningRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    streamRef.current = stream;
    videoRef.current!.srcObject = stream;
    await videoRef.current!.play();

    runningRef.current = true;
    setIsStreaming(true);
    setStatus("กำลังตรวจจับ...");
    requestAnimationFrame(loop);
  }

  function stopCamera() {
    runningRef.current = false;
    setIsStreaming(false);

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
    }

    setEmotion("-");
    setConf(0);
    setStatus("กล้องปิดแล้ว");
  }

  // =====================================================
  // Boot
  // =====================================================
  useEffect(() => {
    (async () => {
      try {
        await loadOpenCV();
        await loadCascade();
        await loadModel();
        setStatus("พร้อมใช้งาน (กด OPEN_CAM)");
      } catch (e: any) {
        setStatus(`โหลดไม่สำเร็จ: ${e.message}`);
      }
    })();
  }, []);

  // =====================================================
  // UI
  // =====================================================
  return (
    <div className="app-root">
      <div className="app-card">
        {/* Header */}
        <div className="app-header">
          <h1 className="app-title">Face Emotion AI</h1>
          <span>{status}</span>
        </div>

        {/* Status Panel */}
        <div className="status-panel">
          <div className="status-box">
            <div className="status-label">EMOTION</div>
            <div className="status-value">{emotion}</div>
          </div>

          <div className="status-box">
            <div className="status-label">CONFIDENCE</div>
            <div className="status-value">
              {(confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        {/* Controls */}
        <div className="controls">
          {!isStreaming ? (
            <button
              className="control-btn"
              onClick={startCamera}
            >
              OPEN_CAM
            </button>
          ) : (
            <button
              className="control-btn stop"
              onClick={stopCamera}
            >
              STOP_CAM
            </button>
          )}
        </div>
        {/* Camera */}
        <div className="camera-wrap">
          <canvas ref={canvasRef} className="camera-canvas" />
          <video
            ref={videoRef}
            className="hidden"
            playsInline
            muted
          />
        </div>

        
      </div>
    </div>
  );
}