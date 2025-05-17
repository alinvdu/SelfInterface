import { useState, useEffect } from "react";

// assets = { images: [...], videos: [...] }
function usePreloadAssets({ images = [], videos = [] }) {
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let isCancelled = false;
    let total = images.length + videos.length;
    if (total === 0) {
      setLoaded(true);
      return;
    }
    let loadedCount = 0;

    const onAssetLoaded = () => {
      loadedCount++;
      if (loadedCount === total && !isCancelled) setLoaded(true);
    };

    // Preload images
    images.forEach((src) => {
      const img = new window.Image();
      img.onload = img.onerror = onAssetLoaded;
      img.src = src;
    });

    // Preload videos
    videos.forEach((src) => {
      const vid = document.createElement("video");
      vid.preload = "auto";
      // Try to load enough so it will play smoothly:
      const done = () => {
        // Remove listeners after fired
        vid.removeEventListener("canplaythrough", done);
        vid.removeEventListener("error", done);
        onAssetLoaded();
      };
      vid.addEventListener("canplaythrough", done);
      vid.addEventListener("error", done);
      vid.src = src;
      // For some browsers, force a tiny load
      vid.load();
    });

    return () => {
      isCancelled = true;
    };
  }, [images, videos]);

  return loaded;
}
export default usePreloadAssets;
