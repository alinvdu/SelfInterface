import React, { useState, useEffect } from 'react';

const LoadingDiv = ({ 
  isLoading = true,
  isGlowing = false,
  children, 
  duration = 2,
  width = '100%',
  height = '100%',
  borderWidth = 3,
  loadingColor = '#FFFFFF',
  borderColor = 'rgba(255, 255, 255, 0.2)',
  borderRadius = '8px',
  backgroundColor = 'transparent',
  loadingSegmentPercentage = 25,
  glowColor = '#FFFFFF',
  glowPulseDuration = 1.5,
}) => {
  const [position, setPosition] = useState(0);
  const [glowPulse, setGlowPulse] = useState(0);
  const containerRef = React.useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  
  // Format width and height to CSS values
  const formatSize = (size) => {
    if (typeof size === 'number') return `${size}px`;
    return size;
  };
  
  // Update dimensions when the container size changes
  useEffect(() => {
    if (!containerRef.current) return;
    
    const updateDimensions = () => {
      if (containerRef.current) {
        const { offsetWidth, offsetHeight } = containerRef.current;
        setDimensions({ width: offsetWidth, height: offsetHeight });
      }
    };
    
    updateDimensions();
    
    const observer = new ResizeObserver(updateDimensions);
    observer.observe(containerRef.current);
    
    return () => {
      if (containerRef.current) {
        observer.unobserve(containerRef.current);
      }
    };
  }, []);
  
  // Animation effect for loading
  useEffect(() => {
    if (!isLoading) return;
    
    let animationFrame;
    let startTime = null;
    
    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = ((timestamp - startTime) / (duration * 1000)) % 1;
      setPosition(progress);
      animationFrame = requestAnimationFrame(animate);
    };
    
    animationFrame = requestAnimationFrame(animate);
    
    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [isLoading, duration]);
  
  // Animation effect for glow pulsation
  useEffect(() => {
    if (!isGlowing || isLoading) return;
    
    let animationFrame;
    let startTime = null;
    
    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      // Use sine wave for smooth pulsation
      const progress = ((timestamp - startTime) / (glowPulseDuration * 1000)) % 1;
      const pulseValue = Math.sin(progress * Math.PI * 2) * 0.5 + 0.5; // Normalize to 0-1
      setGlowPulse(pulseValue);
      animationFrame = requestAnimationFrame(animate);
    };
    
    animationFrame = requestAnimationFrame(animate);
    
    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [isGlowing, isLoading, glowPulseDuration]);
  
  // Calculate the SVG path for the border
  const calculateSVGPath = () => {
    if (dimensions.width === 0 || dimensions.height === 0) return null;
    
    // Determine actual container size (accounting for border width)
    const actualWidth = dimensions.width - 2 * borderWidth;
    const actualHeight = dimensions.height - 2 * borderWidth;
    
    // Calculate proper radius
    let radiusValue;
    if (typeof borderRadius === 'string' && borderRadius.includes('%')) {
      const percentage = parseFloat(borderRadius) / 100;
      const maxRadius = Math.min(actualWidth, actualHeight) / 2;
      radiusValue = maxRadius * Math.min(percentage * 2, 1);
    } else {
      radiusValue = parseFloat(borderRadius);
      radiusValue = Math.min(radiusValue, Math.min(actualWidth, actualHeight) / 2);
    }
    
    // For perfect circles (check if radius is very close to making a circle)
    const isCircle = radiusValue >= Math.min(actualWidth, actualHeight) / 2 - 1;
    
    if (isCircle) {
      // Use a perfect circle path
      const cx = actualWidth / 2;
      const cy = actualHeight / 2;
      const r = Math.min(actualWidth, actualHeight) / 2;
      
      // Perfect circle path
      return `M ${cx + r} ${cy} A ${r} ${r} 0 1 1 ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`;
    }
    
    // For rounded rectangles
    return `
      M ${radiusValue} 0
      H ${actualWidth - radiusValue}
      C ${actualWidth} 0, ${actualWidth} 0, ${actualWidth} ${radiusValue}
      V ${actualHeight - radiusValue}
      C ${actualWidth} ${actualHeight}, ${actualWidth} ${actualHeight}, ${actualWidth - radiusValue} ${actualHeight}
      H ${radiusValue}
      C 0 ${actualHeight}, 0 ${actualHeight}, 0 ${actualHeight - radiusValue}
      V ${radiusValue}
      C 0 0, 0 0, ${radiusValue} 0
      Z
    `;
  };
  
  // Calculate the total path length (perimeter)
  const [pathLength, setPathLength] = useState(0);
  
  useEffect(() => {
    if (containerRef.current) {
      const path = containerRef.current.querySelector('path');
      if (path) {
        setPathLength(path.getTotalLength());
      }
    }
  }, [dimensions]);
  
  // Calculate dash array and offset based on position
  const calculateDashProps = () => {
    if (pathLength === 0) return { strokeDasharray: '0', strokeDashoffset: '0' };
    
    const segmentLength = (loadingSegmentPercentage / 100) * pathLength;
    const dashArray = `${segmentLength} ${pathLength - segmentLength}`;
    const dashOffset = pathLength - (position * pathLength);
    
    return {
      strokeDasharray: dashArray,
      strokeDashoffset: dashOffset
    };
  };
  
  const svgPath = calculateSVGPath();
  const dashProps = calculateDashProps();
  
  // Styles for containers - using plain CSS instead of Tailwind
  const outerContainerStyle = {
    width: formatSize(width),
    height: formatSize(height),
    minWidth: formatSize(width),
    minHeight: formatSize(height),
    position: 'relative'
  };
  
  const mainContainerStyle = {
    position: 'relative',
    width: '100%',
    height: '100%',
    backgroundColor,
    borderRadius,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  };
  
  const svgContainerStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    zIndex: 0
  };
  
  const svgStyle = {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    overflow: 'visible'
  };
  
  const contentWrapperStyle = {
    position: 'relative',
    zIndex: 1,
    padding: `${borderWidth}px`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 'auto',
    height: 'auto'
  };
  
  // Generate unique IDs for filters
  const baseFilterId = `base-filter-${Math.random().toString(36).substring(2, 9)}`;
  const pulseFilterId = `pulse-filter-${Math.random().toString(36).substring(2, 9)}`;
  
  // Calculate dynamic properties based on pulse
  const getStrokeWidth = () => {
    const minWidth = borderWidth;
    const maxWidth = borderWidth * 1.8;
    return minWidth + (maxWidth - minWidth) * glowPulse;
  };
  
  const getGlowStrength = () => {
    const minStrength = 3;
    const maxStrength = 8;
    return minStrength + (maxStrength - minStrength) * glowPulse;
  };
  
  // Allow customization of the opacity range
  const getOpacity = () => {
    const minOpacity = 0.6;
    const maxOpacity = 1;
    return minOpacity + (maxOpacity - minOpacity) * glowPulse;
  };
  
  // Dynamic values
  const dynamicStrokeWidth = getStrokeWidth();
  const dynamicGlowStrength = getGlowStrength();
  const dynamicOpacity = getOpacity();
  
  return (
    <div style={outerContainerStyle}>
      <div
        ref={containerRef}
        style={mainContainerStyle}
      >
        {/* Border SVG - positioned behind content */}
        <div style={svgContainerStyle}>
          {svgPath && (
            <svg 
              style={svgStyle}
              viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* SVG filters for glow effects */}
              <defs>
                {/* Base filter */}
                <filter id={baseFilterId} x="-10%" y="-10%" width="120%" height="120%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                
                {/* Pulse filter - changes dynamically */}
                <filter id={pulseFilterId} x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation={dynamicGlowStrength} result="blur" />
                  <feFlood floodColor={glowColor} floodOpacity={dynamicOpacity} result="colored-blur" />
                  <feComposite in="colored-blur" in2="blur" operator="in" result="colored-blur" />
                  <feComposite in="colored-blur" in2="SourceGraphic" operator="over" />
                </filter>
              </defs>
              
              {/* Background border */}
              {!isLoading && isGlowing ? null : (
                <path
                  d={svgPath}
                  fill="none"
                  stroke={borderColor}
                  strokeWidth={borderWidth}
                  strokeLinecap="round"
                  transform={`translate(${borderWidth}, ${borderWidth})`}
                />
              )}
              
              {/* Glow effect layers - only visible when isGlowing=true and isLoading=false */}
              {!isLoading && isGlowing && (
                <>
                  {/* Outer glow layer - changes with pulse */}
                  <path
                    d={svgPath}
                    fill="none"
                    stroke={glowColor}
                    strokeWidth={dynamicStrokeWidth * 1.2}
                    strokeOpacity={dynamicOpacity * 0.8}
                    strokeLinecap="round"
                    transform={`translate(${borderWidth}, ${borderWidth})`}
                    filter={`url(#${baseFilterId})`}
                  />
                  
                  {/* Primary glow layer - pulses */}
                  <path
                    d={svgPath}
                    fill="none"
                    stroke={glowColor}
                    strokeWidth={dynamicStrokeWidth}
                    strokeOpacity={dynamicOpacity}
                    strokeLinecap="round"
                    transform={`translate(${borderWidth}, ${borderWidth})`}
                    filter={`url(#${pulseFilterId})`}
                  />
                  
                  {/* Sharp visible border on top */}
                  <path
                    d={svgPath}
                    fill="none"
                    stroke={glowColor}
                    strokeWidth={borderWidth}
                    strokeOpacity={dynamicOpacity}
                    strokeLinecap="round"
                    transform={`translate(${borderWidth}, ${borderWidth})`}
                  />
                </>
              )}
              
              {/* Loading indicator */}
              {isLoading && (
                <path
                  d={svgPath}
                  fill="none"
                  stroke={loadingColor}
                  strokeWidth={borderWidth}
                  strokeLinecap="round"
                  transform={`translate(${borderWidth}, ${borderWidth})`}
                  {...dashProps}
                />
              )}
            </svg>
          )}
        </div>
        
        {/* Content wrapper - positioned in front of SVG */}
        <div style={contentWrapperStyle}>
          {children}
        </div>
      </div>
    </div>
  );
};

export default LoadingDiv;
