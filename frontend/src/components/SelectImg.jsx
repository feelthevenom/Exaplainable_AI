import { useState, useRef } from "react";
import "./preview.css"
import Button from '@mui/material/Button';

export default function SelectImg({btnvis}) {
	// FIles States
	const [imagePreview, setImagePreview] = useState(null);
	const [videoPreview, setVideoPreview] = useState(null);

	// FIle Picker Ref because we are not useing the standard File picker input
	const filePicekerRef = useRef(null);

	function previewFile(e) {
		// Reading New File (open file Picker Box)
		const reader = new FileReader();

		// Gettting Selected File (user can select multiple but we are choosing only one)
		const selectedFile = e.target.files[0];
		if (selectedFile) {
			reader.readAsDataURL(selectedFile);
		}

		// As the File loaded then set the stage as per the file type
		reader.onload = (readerEvent) => {
			if (selectedFile.type.includes("image")) {
				setImagePreview(readerEvent.target.result);
			} else if (selectedFile.type.includes("video")) {
				setVideoPreview(readerEvent.target.result);
			}
		};
	}

	function clearFiles() {
		setImagePreview(null);
		setVideoPreview(null);
	}

	return (
		<div>
			<h1>Preview Image</h1>

			<div className="btn-container" style={{display:btnvis}}>
				<input
					ref={filePicekerRef}
					accept="image/*, video/*"
					onChange={previewFile}
					type="file"
					hidden
				/>
				<button className="btn" onClick={() => filePicekerRef.current.click()}>
					Choose
				</button>
				{(imagePreview || videoPreview) && (
					<button className="btn" onClick={clearFiles}>
						x
					</button>
				)}
			</div>

			<div className="preview">
				{imagePreview != null && <img src={imagePreview} alt="" />}
				{videoPreview != null && <video controls src={videoPreview}></video>}
			</div>
		<Button id='btn' variant="contained" color="success" size='large' >
        Explain
      	</Button>
		</div>
	);
}
