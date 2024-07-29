import * as React from 'react';
import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import CloudUpload from '@mui/icons-material/CloudUpload';
import axios from 'axios';


const InputFileUpload = () => {
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('File uploaded successfully:', response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <Button
      component="label"
      variant="contained"
      startIcon={<CloudUpload />}
    >
      Upload file
      <input
        type="file"
        hidden
        onChange={handleFileChange}
      />
    </Button>
  );
};

export default InputFileUpload;
