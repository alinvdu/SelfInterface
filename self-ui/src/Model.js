import React, { useEffect, useRef } from 'react';
import { useGLTF, useAnimations } from '@react-three/drei';

function Model({ isPlaying }) {
  const { scene, animations } = useGLTF('/assets/ai-psychologist6.glb');
  const { actions } = useAnimations(animations, scene);
  const talkActionRef = useRef(null);

  useEffect(() => {
    if (animations.length) {
      // Assume the first clip is your idle animation
      const idleClip = animations[0];
      const idleAction = actions[idleClip.name];
      idleAction.reset().play();

      // Setup the talk animation (assume it's the second clip)
      if (animations[1]) {
        const talkClip = animations[1];
        const talkAction = actions[talkClip.name];
        talkAction.timeScale = 1.65;
        talkActionRef.current = talkAction;
        talkAction.reset(); // start with talk animation stopped
        if (isPlaying) {
          talkAction.play();
        }
      }
    }
  }, [actions, animations]);

  useEffect(() => {
    if (talkActionRef.current) {
      if (isPlaying) {
        // When audio is playing, start or restart the talk animation
        talkActionRef.current.reset().play();
      } else {
        // When audio stops, stop the talk animation
        talkActionRef.current.stop();
        talkActionRef.current.reset();
      }
    }
  }, [isPlaying]);

  return (
    <group position={[0, -1.2, 0]}>
      <primitive object={scene} dispose={null} />
    </group>
  );
}

export default Model;
