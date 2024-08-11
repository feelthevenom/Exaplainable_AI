import React, { useState, useEffect } from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Unstable_Grid2';
import { Typography } from '@mui/material';
import "./styles.css";

import SelectImg from "./SelectImg"
import Input from "./Input"

const Item = styled(Paper)(({ theme, Iheight }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? 'dark' : '#fff',
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  height: Iheight
}));

const Tool = styled(Paper)(({ theme, Iheight }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? 'dark' : '#fff',
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: 'center',
  justifyContent: "space-evenly",
  color: theme.palette.text.secondary,
  height: Iheight
}));

export default function Main() {
  const [model, setModel] = useState(null);
  const [layer, setLayer] = useState('');
  const [hardware, setHardware] = useState('');
  const [btnvis, setBtnvis] = useState('');

  useEffect(() => {
    if (model && layer && hardware) {
      setBtnvis('none');
    } else {
      setBtnvis('');
    }
  }, [model, layer, hardware]);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={2}>
        <Grid xs={12}>
          <Item Iheight={80}>
            <Typography variant='h3' fontStyle={"italic"} fontFamily={"fantasy"}>
              Explainable AI
            </Typography>
          </Item>
        </Grid>
        <Grid xs={12}>
          <Tool Iheight={200}>
            <Typography variant='h3'>
              <Input
                setModel={setModel}
                setLayer={setLayer}
                setHardware={setHardware}
              />
            </Typography>
          </Tool>
        </Grid>
        <Grid xs={6}>
          <Item Iheight={600}>
            <SelectImg btnvis={btnvis} />
          </Item>
        </Grid>
        <Grid xs={6}>
          <Item Iheight={600}>xs=4</Item>
        </Grid>
      </Grid>
    </Box>
  );
}