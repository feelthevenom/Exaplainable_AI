import React from "react";
import img from "./solar.jpg"
import "./output.css"



export default  function Output(){
    return <div>
    <div >
        {/* <image id="outimg" src = {img} alt="Selected Image"/> */}
        <img src={img} id="outimg" alt="Selected Image" />
        </div>
    </div>
}