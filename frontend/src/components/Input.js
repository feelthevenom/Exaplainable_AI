import React, { useState, useEffect } from 'react';
import { Grid, Box } from '@mui/material';
import InputFileUpload from './FileDropzone';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import CustomizedRadios from "./hardware";

const layers = [
  { value: 'Encoder', label: 'Encoder' },
  { value: 'Decoder', label: 'Decoder' },
  { value: 'Bridge', label: 'Bridge' },
  { value: 'Sigmoid', label: 'Sigmoid' },
];

const models = [
  { value:'unet' , label : 'UNet'},
  { value: 'unetpp' , label : 'UNet ++'},
  { value : 'deeplab' , label : 'DeepLab'},
]
function Input({ setModel, setLayer, setHardware }) {
  const [selectedLayer, setSelectedLayer] = useState('');
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedHardware, setSelectedHardware] = useState('');

  useEffect(() => {
    setModel(selectedModel);
    setLayer(selectedLayer);
    setHardware(selectedHardware);
  }, [selectedLayer, selectedHardware, setModel, setLayer, setHardware]);

  return (
    <Box sx={{ flexGrow: 1, p: 2 }}>
      <Grid container spacing={30} alignItems="center">
        <Grid item>
        <TextField
            id="outlined-select-currency"
            select
            label="Select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            helperText="Please select target layer"
            sx={{ minWidth: 200 }}
          >
            {models.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}

          </TextField>
        </Grid>
        <Grid item>
          <TextField
            id="outlined-select-currency"
            select
            label="Select"
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(e.target.value)}
            helperText="Please select target layer"
            sx={{ minWidth: 200 }}
          >
            {layers.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}

          </TextField>
        </Grid>
        <Grid item>
          <CustomizedRadios onChange={setSelectedHardware} />
        </Grid>
      </Grid>
    </Box>
  );
}

export default Input;