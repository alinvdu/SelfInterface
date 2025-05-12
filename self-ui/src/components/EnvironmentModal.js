import React, { useState } from "react";
import { IoImageOutline } from "react-icons/io5";
import { IoIosSend } from "react-icons/io";
import { IoCloseCircleOutline } from "react-icons/io5";
import LoadingDots from "./LoadingDots";
import { MdErrorOutline } from "react-icons/md";

const EnvironmentModal = ({ api, token, setShowEnvironmentModal, setSelectedEnv, setEnvironments, environments }) => {
    const [prompt, setPrompt] = useState(null);
    const [name, setName] = useState("");
    const [imageUrl, setImageUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const onApply = (imageUrl) => {
        setSelectedEnv({
            name,
            url: imageUrl
        })
    }

    const generate = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${api}/generate_environment`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify({ prompt }),
            });
            const { imageBase64 } = await res.json();
            setImageUrl(imageBase64);
        } catch(err) {
            console.error(err)
        }

        setLoading(false);
    };

    const save = async (apply = false) => {
        setLoading(true);
        const res = await fetch(`${api}/save_environment`, {
            method: "POST",
            headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`
            },
            body: JSON.stringify({ name, imageUrl }),
        });
        const { fullImage, thumbnail } = await res.json();

        setEnvironments(prev => [...prev, {
            fullImage,
            thumbnail,
            name
        }])

        if (apply) onApply(fullImage);
        setShowEnvironmentModal(false);
        setLoading(false);
    };

    return (
        <div style={{
            position: 'absolute', inset: 0, zIndex: 999,
            background: 'rgba(3,3,8,0.65)',
            backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
        <div
            style={{
                position: 'relative', width: 620, height: imageUrl ? 520 : 200,
                borderRadius: 45,
                background: 'rgba(20,20,20,0.95)',
                backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
                border: '1px solid rgba(255,255,255,0.15)',
                overflow: 'hidden', boxShadow: '0 8px 40px rgba(0,0,0,0.5)',
                display: "flex",
                alignItems: "center",
                justifyContent: "flex-start",
                flexDirection: "column",
                padding: 25,
                boxSizing: "border-box",
            }}
        >
            {imageUrl && <div onClick={() => {
                setImageUrl(null);
            }} style={{
                position: "absolute",
                top: 330,
                right: 35,
                borderRadius: 5,
                fontSize: 14,
                cursor: "pointer",
                padding: "6px 12px",
                background: 'rgba(20, 20, 20, 0.75)',
            }}>
                <span>Clear picture</span>
            </div>}
            {error ? <div style={{
                position: "absolute",
                top: 80,
                background: 'rgba(20, 20, 20, 0.75)',
                left: 0,
                left: "50%",
                transform: "translateX(-50%)",
                padding: "8px 12px",
                borderRadius: 5,
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "center"
            }}>
                <MdErrorOutline style={{
                    fontSize: 25,
                    color: "red"
                }} />
                <span>{error}</span>
            </div> : null}
            <div style={{
                display: "flex",
                width: "100%",
                flexDirection: "column",
            }}>
            <div style={{
                display: "flex",
                width: "100%",
                flexDirection: "column",
                maxHeight: 50
            }}>
            <span style={{
                fontSize: 18,
                textAlign: "left",
                opacity: 0.85
            }}>Generate Environment</span>
            <div style={{
                marginTop: 10,
                width: "100%",
                height: 1,
                background: "rgba(255, 255, 255, 0.25)"
            }} />
            </div>
            {imageUrl && (
                <div style={{
                    marginTop: 10,
                    borderRadius: 8,
                    width: "100%",
                    border: '1px solid rgba(255, 255, 255, 0.35)',
                    height: 300,
                    backgroundSize: "cover",
                    backgroundImage: `url("data:image/png;base64,${imageUrl}")`
                }} />
            )}
            </div>
            <div style={{
                marginTop: 25,
                display: "flex",
                width: "100%",
                alignItems: "center"
            }}>
            {imageUrl ?
            <div style={{
                display: "flex",
                flexDirection: "column",
                width: "100%"
            }}>
                <input
                    style={{
                        borderRadius: 15,
                        border: error ? "1px solid red" : '1px solid rgba(255,255,255,0.15)',
                        background: 'rgba(30,30,30,0.95)',
                        color: "white",
                        padding: 15,
                    }}
                    type="text"
                    value={name}
                    onChange={e => {
                        setError(null)
                        setName(e.target.value)
                    }}
                    placeholder="Name your environment..."
                />
                <div style={{
                    display: "flex",
                    marginTop: 15,
                    flexDirection: "center",
                    justifyContent: "center",
                    fontSize: 15
                }}>
                    <div style={{
                        padding: '8px 12px',
                        borderRadius: 8,
                        backgroundColor: 'rgba(50, 50, 50, 0.35)',
                        border: '1px solid rgba(255, 255, 255, 0.25)',
                        marginRight: 10,
                        cursor: "pointer"
                    }} onClick={() => {
                        if (!name) {
                            setError("Please provide a name for your environment")
                            return
                        }

                        if (environments.find(env => env.name === name)) {
                            setError("Name already used, pick another one")
                            return
                        }

                        if (error) {
                            return
                        }
                        save(false)
                    }}>Save</div>
                    <div style={{
                        padding: '8px 12px',
                        borderRadius: 8,
                        backgroundColor: 'rgba(50, 50, 50, 0.35)',
                        border: '1px solid rgba(255, 255, 255, 0.25)',
                        marginRight: 10,
                        cursor: "pointer"
                    }} onClick={() => {
                        if (!name) {
                            setError("Please provide a name for your environment")
                            return
                        }

                        if (error) {
                            return
                        }
                        save(true)
                    }}>Save & Apply</div>
                </div>
            </div> :
            <div style={{
                position: "relative",
                width: "100%",
                height: "100%",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
            }}>
            <textarea value={prompt} type="text" onChange={(e) => {
                setPrompt(e.target.value)
            }} onKeyDown={(e) => {
                if (loading) {
                    return
                }

                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  generate(); 
                }
              }} style={{
                flex: 1,
                height: 100,
                borderRadius: 25,
                border: '1px solid rgba(255,255,255,0.15)',
                background: 'rgba(30,30,30,0.95)',
                color: "white",
                padding: 18,
                boxSizing: "border-box",
                resize: "none"
            }} placeholder="Describe your environment here, enter or click send to generate..." />
            <IoIosSend onClick={() => {
                if (loading) {
                    return
                }

                generate()
            }} style={{
                fontSize: 25,
                opacity: 0.85,
                color: loading ? "grey" : "white",
                marginLeft: 15,
                cursor: "pointer"
            }} />
            </div>}
            {loading && <div style={{
                position: "absolute",
                top: 60,
                right: 0,
                bottom: 0,
                left: 0,
                background: 'rgba(20, 20, 20, 0.85)',
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexDirection: "column"
            }}>
                <div
                >{!imageUrl ? "Generating image, please wait" : "Saving image, please wait"}</div>
                <LoadingDots size={7} extraStyles={{
                    height: 40
                }} />
            </div>}
            <IoCloseCircleOutline style={{
                position: "absolute",
                top: 25,
                right: 25,
                fontSize: 23,
                opacity: 0.85,
                cursor: "pointer",
                color: "white"
            }} onClick={() => {
            setShowEnvironmentModal(false);
            }} />
            </div>
        </div>
    </div>);
}

export default EnvironmentModal;