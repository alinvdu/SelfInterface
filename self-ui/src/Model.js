import React, { useEffect, useRef, useState } from 'react';
import { useGLTF, useAnimations } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

window.messageINeed = []

function Model({ 
  scale = 0.1, 
  visemeSequence = null,
  assistantTalking,
  isPlaying = false,
  currentEmote = null,
  setCurrentEmote = () => {},
  introPosition = false,
  onLoad
}) {
  const { scene, animations } = useGLTF('/assets/ai-modern-psychologist.glb');
  const { actions } = useAnimations(animations, scene);
  const [isAnimating, setIsAnimating] = useState(false);
  const clockRef = useRef(new THREE.Clock());
  const animationTimeRef = useRef(0);
  const initialized = useRef(false);
  const lastVisemeRef = useRef(null);
  const currentShapeKeyValuesRef = useRef({});
  const jawRootRef = useRef(null);
  const stopAnimationTimeout = useRef(null);
  const restartIntervalAfterEmote = useRef(null);
  const prevEmoteRef = useRef(null);

  const currentAnimationRef = useRef(null);
  const availableAnimationsRef = useRef([]);
  const allAnimationsRef = useRef([]);
  const intervalRef = useRef(null);

  const isPlayingRef = useRef(isPlaying);
  const isPausedRef = useRef(false);
  
  // Initialize jaw movement quaternion sequence
  const jawMovementVectorRef = useRef([]);
  // Keep track of current index in the quaternion array
  const currentJawIndexRef = useRef(0);
  // Keep track of target index in the quaternion array
  const targetJawIndexRef = useRef(0);
  // Track movement speed - how many indices to move per frame
  const jawMovementSpeedRef = useRef(1);

  useEffect(() => {
    // Call onLoad when the scene and animations are loaded and initialization is complete
    if (onLoad && scene && animations.length > 0) {
      onLoad();
    }
  }, [scene, animations, onLoad]);
  
  // Initialize the jaw movement vector with the sequence data from the pasted quaternions
  const initializeJawMovementVector = () => {
    // First entry is the default quaternion
    const defaultQuaternion = new THREE.Quaternion(
      -0.000059061741922050714,
      -0.000042112773371627554,
      0.7070924043655396,
      0.7071211338043213
    );
    
    // Create the full sequence array starting with default
    const sequence = [defaultQuaternion];
    
    // Add the quaternion sequence (107 entries) - using just a subset for efficiency
    // We'll use this to represent the full range of jaw motion
    const quaternionData = [
      { x: -0.00005864655017964742, y: -0.00004233081482044212, z: 0.7080549711853324, w: 0.7061572954530596 },
      { x: -0.00005876383784621292, y: -0.00004217710349479283, z: 0.7099617837592365, w: 0.7042401845310273 },
      { x: -0.000058883540682937, y: -0.00004201932337090428, z: 0.7119097215135262, w: 0.7022709759732415 },
      { x: -0.000059002791736294176, y: -0.00004186122085429246, z: 0.7138521971474971, w: 0.7002963792500428 },
      { x: -0.000059118766827064885, y: -0.00004170657287391537, z: 0.7157431405080742, w: 0.6983636139724172 },
      { x: -0.000059238528642203266, y: -0.00004154594517988542, z: 0.7176977407158546, w: 0.6963547482505519 },
      { x: -0.00005935362253588027, y: -0.00004139067736539193, z: 0.7195780117819408, w: 0.694411598974712 },
      { x: -0.000059471073621842774, y: -0.000041231308751769145, z: 0.7214986877723988, w: 0.6924157914966818 },
      { x: -0.00005958528813589078, y: -0.00004107542948123037, z: 0.7233682939383378, w: 0.6904623788023936 },
      { x: -0.000059704606022717045, y: -0.0000409116237093544, z: 0.7253234179881952, w: 0.6884082542358272 },
      { x: -0.00005981654774649742, y: -0.00004075704062377061, z: 0.7271595379701004, w: 0.6864684923714858 },
      { x: -0.000059932184703295174, y: -0.00004059642512914903, z: 0.7290581827298174, w: 0.6844517147848816 },
      { x: -0.00006004599338270723, y: -0.00004043741578333763, z: 0.7309287304321735, w: 0.6824537883507138 },
      { x: -0.0000601675504459305, y: -0.00004026444819032403, z: 0.7329388679455667, w: 0.6802944926716962 },
      { x: -0.000060310339705167794, y: -0.000040050148088734564, z: 0.7353497589041564, w: 0.6776877709897814 },
      { x: -0.00006046727471591235, y: -0.00003981270274235139, z: 0.7380036406133432, w: 0.674796718668928 },
      { x: -0.000060636236654784975, y: -0.00003955477845848373, z: 0.7408658514397294, w: 0.6716530188328029 },
      { x: -0.00006078570947931203, y: -0.00003932459593613931, z: 0.7434022704736383, w: 0.6688445664120787 },
      { x: -0.000060939798848172474, y: -0.00003908529334532222, z: 0.7460213856103044, w: 0.6659219806638839 },
      { x: -0.00006109113197318373, y: -0.00003884824832496778, z: 0.7485980356572546, w: 0.6630241138155362 },
      { x: -0.0000612415464870592, y: -0.0000386106191536654, z: 0.7511634292345706, w: 0.6601162772475411 },
      { x: -0.00006139283563274866, y: -0.00003836953588271361, z: 0.7537482310184229, w: 0.6571633005829163 },
      { x: -0.00006154317925202848, y: -0.00003812786167309625, z: 0.7563214241370114, w: 0.6542002027840377 },
      { x: -0.00006169080209209798, y: -0.00003788848775051846, z: 0.7588525430441457, w: 0.6512624834853695 },
      { x: -0.0000618374973038825, y: -0.00003764854410876503, z: 0.7613722512861698, w: 0.6483149713248783 },
      { x: -0.00006197801027532684, y: -0.00003741673729934666, z: 0.7637900508397069, w: 0.6454647688687233 },
      { x: -0.00006212983532223315, y: -0.000037164054208739194, z: 0.7664072958193306, w: 0.6423549403057973 },
      { x: -0.00006217610756990167, y: -0.00003709469251013859, z: 0.7671310660431555, w: 0.6414904039716356 },
      { x: -0.0000622010438946266, y: -0.00003706289478344692, z: 0.7674690964482308, w: 0.6410859496852653 },
      { x: -0.00006222566640097215, y: -0.00003703146549689867, z: 0.7678028934529253, w: 0.6406861355558748 },
      { x: -0.00006225056839548282, y: -0.000036999647327046054, z: 0.76814050026554, w: 0.6402813276194708 },
      { x: -0.00006227426851718498, y: -0.00003696933476030763, z: 0.7684618323473709, w: 0.6398956310262429 },
      { x: -0.00006229943271811824, y: -0.000036937117460401244, z: 0.7688030357037279, w: 0.6394856518841564 },
      { x: -0.00006232428347495183, y: -0.000036905268874244985, z: 0.7691400102448515, w: 0.6390803174738611 },
      { x: -0.00006234882139845147, y: -0.000036873789376702334, z: 0.769472763499576, w: 0.6386796342453478 },
      { x: -0.00006237363775708876, y: -0.00003684192043437919, z: 0.769809313526493, w: 0.6382739472919944 },
      { x: -0.00006239637087755428, y: -0.000036812698253949194, z: 0.770117630018864, w: 0.6379019119530217 },
      { x: -0.00006242292361165533, y: -0.00003677853157674727, z: 0.770477772281779, w: 0.63746687688585 },
      { x: -0.00006244739350398559, y: -0.000036747011920624677, z: 0.7708096858861959, w: 0.637065497844547 },
      { x: -0.0000624712574445255, y: -0.00003671624217319865, z: 0.77113340018057, w: 0.6366736241380941 },
      { x: -0.0000624704219613084, y: -0.00003671980017546662, z: 0.7711027733964585, w: 0.6367107172934121 },
      { x: -0.00006244276077258041, y: -0.00003676031093032977, z: 0.7706897869579276, w: 0.6372105399283264 },
      { x: -0.00006241473552154709, y: -0.000036801299982206084, z: 0.7702714341796991, w: 0.6377161851804882 },
      { x: -0.00006238634524848344, y: -0.00003684276674086414, z: 0.7698477032483293, w: 0.6382276429086466 },
      { x: -0.00006235792745559995, y: -0.00003688421724746737, z: 0.769423632722293, w: 0.6387388191022851 },
      { x: -0.00006233015974404359, y: -0.00003692466514354084, z: 0.769009331729781, w: 0.6392375526591492 },
      { x: -0.00006230168760363005, y: -0.00003696608347922428, z: 0.7685845906502571, w: 0.6397481718239583 },
      { x: -0.00006227352742354786, y: -0.00003700699272314394, z: 0.7681645730040457, w: 0.6402524350043259 },
      { x: -0.00006224534042356763, y: -0.000037047886028930475, z: 0.767744224525515, w: 0.6407564224414408 },
      { x: -0.00006221678652705356, y: -0.0000370892557798052, z: 0.76731847495743, w: 0.6412662010477556 },
      { x: -0.00006218684348488852, y: -0.00003713257797055447, z: 0.7668720885636631, w: 0.6417999514211572 },
      { x: -0.00006216027789153421, y: -0.00003717096215691336, z: 0.7664761175100614, w: 0.642272788592675 },
      { x: -0.00006213198341101433, y: -0.00003721179134111772, z: 0.7660544435381287, w: 0.6427756675531935 },
      { x: -0.00006209926500739326, y: -0.000037262178470410945, z: 0.7655319011644134, w: 0.6433979130079234 }
    ];
    
    // Convert the data to THREE.Quaternion objects and add to sequence
    quaternionData.forEach(q => {
      sequence.push(new THREE.Quaternion(q.x, q.y, q.z, q.w));
    });
    
    jawMovementVectorRef.current = sequence;
  };

  // Viseme definitions with their bone and shape key targets
  const visemeDefinitions = {
    "rest": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // For rest, we use the very beginning of the vector (0.0)
          vectorPosition: 0.0,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0 }
      }
    },
    "p/m/": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // A slight jaw movement
          vectorPosition: 0,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0 },
        "V_Explosive": { default: 0, target: 0.4 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0.1 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0 }
      }
    },
    "plosive": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Medium jaw movement
          vectorPosition: 0.2,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0.181 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0.383 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0.184 }
      }
    },
    "a/ah/": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Wide open jaw movement
          vectorPosition: 0.68,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0.381 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0.683 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0.384 }
      }
    },
    "/oo/": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Medium-high jaw movement
          vectorPosition: 0.4,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0.1 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0.8 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0 }
      }
    },
    "/ee/": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Medium-high jaw movement
          vectorPosition: 0.3,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 1 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0.9 }
      }
    },
    "/ff/vv": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Medium-low jaw movement
          vectorPosition: 0.05,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0.65 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0 }
      }
    },
    "/ch/": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Medium jaw movement
          vectorPosition: 0.1,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0 },
        "V_Affricate": { default: 0, target: 0.45 },
        "V_Lip_Open": { default: 0, target: 0 }
      }
    },
    "/kk/": {
      bones: {
        "CC_Base_JawRoot": {
          defaultQuaternion: null, // Will be set during initialization
          // Medium-low jaw movement
          vectorPosition: 0.18,
          ref: null
        }
      },
      shapeKeys: {
        "V_Open": { default: 0, target: 0 },
        "V_Explosive": { default: 0, target: 0 },
        "V_Dental_Lip": { default: 0, target: 0 },
        "V_Tight_O": { default: 0, target: 0 },
        "V_Tight": { default: 0, target: 0 },
        "Vwide": { default: 0, target: 0 },
        "V_Affricate": { default: 0, target: 0 },
        "V_Lip_Open": { default: 0, target: 0 }
      }
    }
  };
  
  // Default test viseme sequence if none provided
  const mapSequence = {
    phrase: "Map",
    visemes: [
      { viseme: "p/m/",  start: 0,   end: 150 },
      { viseme: "a/ah/", start: 150, end: 500 },
      { viseme: "p/m/",  start: 500, end: 750 },
      { viseme: "plosive", start: 750, end: 900 },
      { viseme: "rest",  start: 900, end: 1200 }
    ]
  };
  
  const ahSequence = {
    phrase: "Ah",
    visemes: [
      { viseme: "a/ah/", start: 0,   end: 550 },
      { viseme: "rest",  start: 550, end: 1000 }
    ]
  };
  
  const foodSequence = {
    phrase: "Food",
    visemes: [
      { viseme: "/ff/vv", start: 0,   end: 100 },
      { viseme: "/oo/",   start: 100, end: 500 },
      { viseme: "rest",   start: 500, end: 1000 },
    ]
  };
  
  const seeSequence = {
    phrase: "See",
    visemes: [
      { viseme: "/ee/",   start: 0, end: 800 },
      { viseme: "rest",   start: 800, end: 1000 }
    ]
  };
  
  const fiveSequence = {
    phrase: "Five",
    visemes: [
      { viseme: "/ff/vv", start: 0,   end: 150 },
      { viseme: "a/ah/",  start: 150, end: 500 },
      { viseme: "/ff/vv", start: 500, end: 650 },
      { viseme: "rest",   start: 650, end: 1000 }
    ]
  };
  
  const chewSequence = {
    phrase: "Chew",
    visemes: [
      { viseme: "/ch/", start: 0,   end: 150 },
      { viseme: "/oo/",   start: 150, end: 500 },
      { viseme: "rest",   start: 500, end: 1000 }
    ]
  };
  
  const kickSequence = {
    phrase: "Kick",
    visemes: [
      { viseme: "/kk/",   start: 0,   end: 150 },
      { viseme: "a/ah/",  start: 150, end: 500 },
      { viseme: "/kk/",   start: 500, end: 650 },
      { viseme: "rest",   start: 650, end: 1000 }
    ]
  };  
  
  // References
  const sequenceRef = useRef(visemeSequence);
  const visemeDefsRef = useRef(visemeDefinitions);
  const shapeKeyMeshesRef = useRef({});
  const currentVisemeRef = useRef(null);
  
  // Animation control methods
  const startAnimation = () => {

    if (isAnimating) {
      if (stopAnimationTimeout.current) {
        clearTimeout(stopAnimationTimeout.current)
        stopAnimationTimeout.current = null
      }
      stopAnimation();
    }
  
    clockRef.current.start();
    animationTimeRef.current = 0;
    lastVisemeRef.current = null; // Reset last viseme
    setIsAnimating(true);
    
    // Initialize current shape key values to defaults (typically zeros)
    initializeShapeKeyValues();

    if (!isPlayingRef.current) {
      clockRef.current.stop();
      isPausedRef.current = true;
    }
  };

  const stopAnimation = () => {
    clockRef.current.stop();
    setIsAnimating(false);
  };

  const resetAnimation = () => {
    animationTimeRef.current = 0;
    lastVisemeRef.current = null; // Reset last viseme
    
    // Reset current shape key values
    initializeShapeKeyValues();
    
    // Reset jaw quaternion indices to default
    currentJawIndexRef.current = 0;
    targetJawIndexRef.current = 0;
    
    // Set jaw movement speed
    jawMovementSpeedRef.current = 1;
    
    if (isAnimating) {
      clockRef.current.start();
    }
  };
  
  // Initialize shape key values to defaults
  const initializeShapeKeyValues = () => {
    // Initialize current shape key values to all zeros
    Object.keys(shapeKeyMeshesRef.current).forEach(keyName => {
      currentShapeKeyValuesRef.current[keyName] = 0;
    });
  };

  // Set up animations for body movements
  useEffect(() => {
    if (animations.length > 0) {
      // Set up the idle animation (first one) - this always plays
      const idleClip = animations[0];
      const idleAction = actions[idleClip.name];
      idleAction.timeScale = 0.75;
      idleAction.reset().play();

      const avoidAnimations = {
        'Angry': true,
        'Sad': true
      }
      
      // Store all other animations
      availableAnimationsRef.current = animations.slice(1).filter(clip => !avoidAnimations[clip.name]).map(clip => ({
        name: clip.name,
        action: actions[clip.name]
      }));

      allAnimationsRef.current = animations.slice(1).map(clip => ({
        name: clip.name,
        action: actions[clip.name]
      }));
    }
  }, [actions, animations]);

  const findAnimationByName = (animationName) => {
    if (!allAnimationsRef.current || allAnimationsRef.current.length === 0) {
      return null;
    }
    
    // Find animation that contains the search term
    return allAnimationsRef.current.find(anim => 
      anim.name.toLowerCase().includes(animationName.toLowerCase())
    );
  };

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

  useEffect(() => {
    // if (prevAssistantTalkingRef.current === true && assistantTalking === false) {
    //   // Get the current sequence
    //   const sequence = sequenceRef.current;
    //   if (sequence && sequence.visemes && sequence.visemes.length > 0) {
    //     const lastViseme = sequence.visemes[sequence.visemes.length - 1];
    //     const totalDuration = lastViseme.end;
    //     const remainingTime = totalDuration - animationTimeRef.current;
    //     const progressPercentage = ((animationTimeRef.current / totalDuration) * 100).toFixed(2);
        
    //     console.log(`Animation stopped at: ${progressPercentage}% complete`);
    //     console.log(`Time remaining: ${remainingTime.toFixed(2)}ms out of ${totalDuration}ms total`);
        
    //     // Also push to your global array
    //     window.messageINeed.push({
    //       stoppedAt: animationTimeRef.current,
    //       totalDuration,
    //       progressPercentage: parseFloat(progressPercentage),
    //       remainingTime
    //     });
    //   }
    // }
  
    // Clear any existing interval
    if (assistantTalking && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (assistantTalking && restartIntervalAfterEmote.current) {
      clearTimeout(restartIntervalAfterEmote.current)
      restartIntervalAfterEmote.current = null
    }

    if (assistantTalking && prevEmoteRef.current) {
      prevEmoteRef.current = null
    }
    
    // If assistant is talking, stop any current animation
    // (idle animation continues playing in the background)
    if (assistantTalking) {
      if (currentAnimationRef.current) {
        currentAnimationRef.current.fadeOut(0.2);
        currentAnimationRef.current = null;
      }

      startAnimation();
      return;
    }

    if (initialized.current) {
      // Create a temporary "rest only" sequence
      const resetSequence = {
        phrase: "Reset",
        visemes: [
          { viseme: "rest", start: 0, end: 500 }
        ]
      };
      
      // Set this as the current sequence
      sequenceRef.current = resetSequence;
      
      // Reset animation time to start of this sequence
      animationTimeRef.current = 0;
      
      // Explicitly set target jaw position to rest (index 0)
      targetJawIndexRef.current = 0;
      
      // Let this sequence play for a short duration to reset everything
      stopAnimationTimeout.current = setTimeout(() => {
        stopAnimation();
        stopAnimationTimeout.current = null;
        // Now play random animations
        if (!intervalRef.current) {
          intervalRef.current = setInterval(playRandomAnimation, 15000); // Then every 15 seconds
        }
      }, 200); // Just enough time to animate to rest
    } else {
      // If not initialized yet, just stop animation
      stopAnimation();
      resetAnimation();
      
      // Play random animations
      if (!intervalRef.current) {
        intervalRef.current = setInterval(playRandomAnimation, 15000); // Then every 15 seconds
      }
    }
    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [assistantTalking]);

  useEffect(() => {
    // Update the current sequence reference when visemeSequence changes
    if (visemeSequence) {
      sequenceRef.current = visemeSequence;
    }
  }, [visemeSequence]);

  // Apply scale
  useEffect(() => {
    if (scene) {
      scene.scale.set(scale, scale, scale);
    }
  }, [scene, scale]);

  // Initialize bone and shape key references
  useEffect(() => {
    if (initialized.current || !scene) return;
    
    // Initialize the jaw movement vector with quaternion sequence
    initializeJawMovementVector();
    
    // Find bones and store references
    scene.traverse((object) => {
      // Handle bones
      if (object.isBone) {
        if (object.name === "CC_Base_JawRoot") {
          jawRootRef.current = {
            obj: object,
            default: object.quaternion.clone()
          }
        }

        Object.values(visemeDefsRef.current).forEach(viseme => {
          if (viseme.bones && viseme.bones[object.name]) {
            const boneData = viseme.bones[object.name];
            boneData.ref = object;
            
            // Store default quaternion
            if (!boneData.defaultQuaternion) {
              boneData.defaultQuaternion = object.quaternion.clone();
            }
          }
        });
      }

      // Handle morph targets (shape keys)
      if (object.isMesh && object.morphTargetDictionary && object.morphTargetInfluences && object.name === "CC_Base_Body_1") {
        Object.values(visemeDefsRef.current).forEach(viseme => {
          if (!viseme.shapeKeys) return;
          
          Object.entries(viseme.shapeKeys).forEach(([keyName, keyData]) => {
            const index = object.morphTargetDictionary[keyName];
            if (index !== undefined) {
              // Add a reference to the mesh and index
              keyData.ref = { mesh: object, index };
              
              // Also store in a flat structure for easier access
              shapeKeyMeshesRef.current[keyName] = { mesh: object, index };
              
              // Initialize current value for this shape key
              currentShapeKeyValuesRef.current[keyName] = 0;
            }
          });
        });
      }
    });
    
    initialized.current = true;
  }, [scene]);

  // Find the current viseme based on time
  const getCurrentViseme = (timeMs) => {
    const sequence = sequenceRef.current;
    if (!sequence || !sequence.visemes || sequence.visemes.length === 0) {
      return null;
    }
    
    // Find the viseme for the current time
    for (const viseme of sequence.visemes) {
      if (timeMs >= viseme.start && timeMs <= viseme.end) {
        return viseme;
      }
    }
    
    // Check if we're past the end
    const lastViseme = sequence.visemes[sequence.visemes.length - 1];
    if (timeMs > lastViseme.end) {
      // Animation is complete
      return null;
    }
    
    // We're between visemes, find the nearest upcoming viseme
    const upcomingViseme = sequence.visemes.find(v => timeMs < v.start);
    if (upcomingViseme) {
      // We're just slightly before the next viseme, use rest pose
      return { viseme: "rest", start: timeMs, end: upcomingViseme.start };
    }
    
    // Default to rest
    return { viseme: "rest", start: 0, end: 1 };
  };  

  // Get index in quaternion array based on position
  const getIndexFromPosition = (vectorPosition) => {
    const vector = jawMovementVectorRef.current;
    
    if (!vector || vector.length <= 1) {
      return 0;
    }
    
    // Clamp position between 0 and 1
    const position = Math.max(0, Math.min(1, vectorPosition));
    
    // Calculate the index in the vector (direct mapping)
    const index = Math.round(position * (vector.length - 1));
    
    return index;
  };

  // Move toward target quaternion index by updating the current index
  const moveTowardTargetIndex = () => {
    const currentIndex = currentJawIndexRef.current;
    const targetIndex = targetJawIndexRef.current;
    
    // If we're already at the target, do nothing
    if (currentIndex === targetIndex) {
      return currentIndex;
    }
    
    // Move toward target index (step size determined by speed)
    const step = jawMovementSpeedRef.current;
    
    let newIndex;
    if (currentIndex < targetIndex) {
      // Moving up the array
      newIndex = Math.min(targetIndex, currentIndex + step);
    } else {
      // Moving down the array
      newIndex = Math.max(targetIndex, currentIndex - step);
    }
    
    // Update the current index
    currentJawIndexRef.current = newIndex;
    
    return newIndex;
  };

  // Main animation loop
  useFrame(() => {
    if (!initialized.current || !isAnimating || 
        (isPausedRef.current && !isPlayingRef.current) || 
        jawMovementVectorRef.current.length === 0) {
      return;
    }
      
    // Update time in milliseconds
    const deltaMs = clockRef.current.getDelta() * 1000;
    animationTimeRef.current += deltaMs;
    
    // Find current viseme
    const currentViseme = getCurrentViseme(animationTimeRef.current);
    if (currentViseme) {
      // Check if we're transitioning to a new viseme
      const isNewViseme = !lastVisemeRef.current || 
                          currentViseme.viseme !== lastVisemeRef.current.viseme;
                          
      if (isNewViseme) {
        // Store the current shape key values as our starting point for this viseme
        // (This happens only once at the beginning of each viseme)
        Object.keys(shapeKeyMeshesRef.current).forEach(keyName => {
          const { mesh, index } = shapeKeyMeshesRef.current[keyName];
          currentShapeKeyValuesRef.current[keyName] = mesh.morphTargetInfluences[index];
        });
        
        // Update last viseme reference
        lastVisemeRef.current = currentViseme;
        
        // Determine jaw movement speed based on viseme transition
        if (currentViseme.viseme === "rest") {
          // Slower for transitions to rest
          jawMovementSpeedRef.current = 1;
        } else {
          // Faster for active speech
          jawMovementSpeedRef.current = 2;
        }
      }
      
      // Calculate progress through this viseme
      const visemeDuration = currentViseme.end - currentViseme.start;
      const visemeProgress = Math.min(1, Math.max(0, 
        (animationTimeRef.current - currentViseme.start) / visemeDuration
      ));
      
      // Get the viseme definition
      const visemeDef = visemeDefsRef.current[currentViseme.viseme];
      if (!visemeDef) {
        console.warn(`Viseme definition not found: ${currentViseme.viseme}`);
        return;
      }
      
      // Update bones based on progress
      if (visemeDef.bones) {
        Object.entries(visemeDef.bones).forEach(([boneName, bone]) => {
          if (!bone.ref) return;
          
          if (boneName === "CC_Base_JawRoot" && typeof bone.vectorPosition === 'number') {
            // Get target position from viseme definition and apply viseme progress
            const targetPosition = bone.vectorPosition * visemeProgress;
            
            // Convert the target position to a target index in the quaternion array
            targetJawIndexRef.current = getIndexFromPosition(targetPosition);
            
            // Move toward the target (updates currentJawIndexRef)
            const currentIndex = moveTowardTargetIndex();
            
            // Get the quaternion at the current index (no interpolation)
            const quaternion = jawMovementVectorRef.current[currentIndex];
            
            // Apply the quaternion directly
            bone.ref.quaternion.copy(quaternion);
          }
        });
      }
      
      // Update shape keys based on progress
      if (visemeDef.shapeKeys) {
        Object.entries(visemeDef.shapeKeys).forEach(([keyName, shapeKey]) => {
          if (!shapeKey.ref) return;
          
          const { mesh, index } = shapeKey.ref;
          const targetValue = shapeKey.target || 0;
          
          // Get the starting value for this shape key (from when we entered this viseme)
          const startValue = currentShapeKeyValuesRef.current[keyName] || 0;
          
          // Interpolate between the starting value and target value
          const newValue = startValue + (targetValue - startValue) * visemeProgress;
          
          // Apply the new value
          mesh.morphTargetInfluences[index] = newValue;
        });
      }
    } else {
      // No active viseme, reset to rest pose
      resetToRestPose();
      
      // Loop the animation if we've reached the end
      const sequence = sequenceRef.current;
      if (sequence && sequence.visemes && sequence.visemes.length > 0) {
        const lastViseme = sequence.visemes[sequence.visemes.length - 1];
        if (animationTimeRef.current > lastViseme.end + 200) { // Small delay before looping
          resetAnimation();
        }
      }
    }
  });

  // Reset to rest pose
  const resetToRestPose = () => {
    const restViseme = visemeDefsRef.current["rest"];
    if (!restViseme) return;
  
    // Directly set target index to rest position
    targetJawIndexRef.current = 0;
    currentJawIndexRef.current = 0; // immediate reset
  
    // Apply rest quaternion directly to the jaw
    if (restViseme.bones) {
      Object.values(restViseme.bones).forEach(bone => {
        if (!bone.ref) return;
  
        if (bone.ref.name === "CC_Base_JawRoot") {
          const quaternion = jawMovementVectorRef.current[0];
          bone.ref.quaternion.copy(quaternion);
        } else if (bone.defaultQuaternion) {
          bone.ref.quaternion.copy(bone.defaultQuaternion);
        }
      });
    }
  
    // Reset all shape keys to default instantly
    Object.entries(shapeKeyMeshesRef.current).forEach(([keyName, { mesh, index }]) => {
      mesh.morphTargetInfluences[index] = 0;
      currentShapeKeyValuesRef.current[keyName] = 0;
    });
  };  

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  
    if (isPlaying) {
      if (isPausedRef.current) {
        clockRef.current.start();
        isPausedRef.current = false;
      }
    } else {
      clockRef.current.stop();
      isPausedRef.current = true;
      resetToRestPose(); // Explicitly reset pose on pause
    }
  }, [isPlaying]);

  useEffect(() => {
    // Only trigger when currentEmote changes from previous value, 
    // is not null, and assistant is not talking
    if (currentEmote && 
        prevEmoteRef.current !== currentEmote && 
        !assistantTalking) {
      
      // Store current emote in ref to track changes
      prevEmoteRef.current = currentEmote;
      
      // Stop any running interval animations
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      
      // Stop current animation if any
      if (currentAnimationRef.current) {
        currentAnimationRef.current.fadeOut(0.2);
        currentAnimationRef.current = null;
      }

      // Play the appropriate animation based on emote type
      switch (currentEmote) {
        case 'happy': {
          const animation = findAnimationByName('Smile_left_right')
          animation.action.loop = THREE.LoopOnce;
          animation.action.reset().fadeIn(0.3).play();
          currentAnimationRef.current = animation.action;
          break;
        }
        case 'sad': {
          const animation = findAnimationByName('Sad')
          animation.action.loop = THREE.LoopOnce;
          animation.action.reset().fadeIn(0.3).play();
          currentAnimationRef.current = animation.action;
          break;
        }
        case 'anger':
        case 'disappointment': {
          const animation = findAnimationByName('Angry')
          animation.action.loop = THREE.LoopOnce;
          animation.action.reset().fadeIn(0.3).play();
          currentAnimationRef.current = animation.action;
          break;
        }
        default:
          // No specific emotion
          break;
      }

      prevEmoteRef.current = null
      setCurrentEmote(null)
      
      // Restart interval animations after a delay
      restartIntervalAfterEmote.current = setTimeout(() => {
        restartIntervalAfterEmote.current = null
        if (!assistantTalking && !intervalRef.current) {
          intervalRef.current = setInterval(playRandomAnimation, 15000);
        }
      }, 5000); // 5 seconds delay after emotion
    }
  }, [currentEmote, assistantTalking, actions]);
  return (
    <group position={[
      0,
      -0.149, 
      0
    ]}>
      <primitive object={scene} dispose={null} />
    </group>
  );
}

export default Model;