import React, { useEffect, useRef } from 'react';
import { useGLTF, useAnimations } from '@react-three/drei';
import * as THREE from 'three';

function Model({ scale = 0.1, isPlaying, assistantTalking }) {
  const { scene, animations } = useGLTF('/assets/ai-modern-psychologisttalk.glb');
  const { actions } = useAnimations(animations, scene);
  const talkActionRef = useRef(null);
  const idleActionRef = useRef(null);

  const currentAnimationRef = useRef(null);
  const availableAnimationsRef = useRef([]);
  const intervalRef = useRef(null);

  useEffect(() => {
    const playRandomAnimation = () => {
        // Stop current animation if any
        if (currentAnimationRef.current) {
            currentAnimationRef.current.fadeOut(0.5);
            currentAnimationRef.current = null;
        }
        
        // Don't play new animations if assistant is talking or no animations available
        if (assistantTalking || availableAnimationsRef.current.length === 0) {
            return;
        }
        
        // Select random animation from available ones
        const randomIndex = Math.floor(Math.random() * availableAnimationsRef.current.length);
        const randomAnim = availableAnimationsRef.current[randomIndex];
        
        // Configure animation settings
        randomAnim.action.reset();
        randomAnim.action.clampWhenFinished = true; // Stop at the end, don't loop
        randomAnim.action.loop = THREE.LoopOnce;
        randomAnim.action.timeScale = 0.75;
        
        // Fade in the random animation
        randomAnim.action.fadeIn(0.5).play();
        currentAnimationRef.current = randomAnim.action;
    };

    // Clear any existing interval
    if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
    }

    // If assistant is talking, stop any current animation
    if (assistantTalking) {
        if (currentAnimationRef.current) {
            currentAnimationRef.current.fadeOut(0.2);
            currentAnimationRef.current = null;
        }
        return;
    }

    // If not talking, set up interval to play random animations
    playRandomAnimation(); // Play one immediately
    intervalRef.current = setInterval(playRandomAnimation, 15000); // Then every 15 seconds

    // Cleanup function
    return () => {
        if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        }
    };
  }, [assistantTalking]);

  useEffect(() => {
    if (animations.length) {
      // Assume the first clip is your idle animation
      const idleClip = animations[0];
      const idleAction = actions[idleClip.name];
      idleAction.reset().play();
      idleActionRef.current = idleAction;

      availableAnimationsRef.current = animations.slice(1).filter(clip => !clip.name.startsWith("Talk")).map(clip => ({
        name: clip.name,
        action: actions[clip.name]
      }));

      talkActionRef.current = animations.slice(1).filter(clip => clip.name.startsWith("Talk")).map(clip => ({
        name: clip.name,
        action: actions[clip.name]
      }));
    }
  }, [actions, animations]);

  // Handle talk animations based on isPlaying state
  // useEffect(() => {
  //   if (talkActionRef.current && assistantTalking) {
  //     if (isPlaying) {
  //       // When audio is playing, start or restart the talk animation
  //       talkActionRef.current.forEach(clip => {
  //           clip.action.reset().fadeIn(0.3).play();
  //       });
  //     } else {
  //       // When audio stops, stop the talk animation
  //       talkActionRef.current.forEach(clip => {
  //           clip.action.fadeOut(0.3);
  //       });
  //     }
  //   }
  // }, [isPlaying, assistantTalking]);

  // Handle idle animation and talk animations based on assistantTalking state
  useEffect(() => {
    if (idleActionRef.current && talkActionRef.current) {
      if (assistantTalking) {
        // Fade out idle animation when assistant starts talking
        idleActionRef.current.fadeOut(1.5);
        
        // Fade in talk animations
        talkActionRef.current.forEach(clip => {
          clip.action.reset().fadeIn(1.5).play();
        });
      } else {
        // Fade out talk animations
        talkActionRef.current.forEach(clip => {
          clip.action.fadeOut(1.5);
        });
        // Fade back in idle animation when assistant stops talking
        idleActionRef.current.fadeIn(1.5).reset().play();
      }
    }
  }, [assistantTalking]);

  useEffect(() => {
    if (scene) {
      scene.scale.set(scale, scale, scale);
    }
  }, [scene, scale]);

  return (
    <group position={[0, -0.149, 0]}>
      <primitive object={scene} dispose={null} />
    </group>
  );
}

export default Model;
