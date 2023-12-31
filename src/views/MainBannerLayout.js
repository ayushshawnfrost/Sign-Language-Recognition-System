import * as React from 'react';
import PropTypes from 'prop-types';
import { styled } from '@mui/material/styles';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Typography from "../components/Typography";
import Thirsty from '../Thirsty.mp4';
import hungry from '../hungry.mp4';
import brother from '../brother.mp4';
import help from '../help.mp4';
import no from '../no.mp4';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Backdrop from '@mui/material/Backdrop';
import CircularProgress from '@mui/material/CircularProgress';
import Button from '@mui/material/Button';

const MainBannerLayoutRoot = styled('section')(({ theme }) => ({
  color: theme.palette.common.white,
  position: 'relative',
  display: 'block',
  alignItems: 'center',
  [theme.breakpoints.up('xs')]: {
    // height: '120vh',
    // minHeight: 200,
    // maxHeight: 1300,
    paddingTop: "5em"
  },
}));

const Background = styled(Box)(({ opacity }) => ({
  position: 'absolute',
  left: 0,
  right: 0,
  top: 0,
  bottom: 0,
  backgroundSize: 'cover',
  backgroundRepeat: 'no-repeat',
  zIndex: -2,
  opacity: opacity,
}));

function MainBannerLayout(props) {
  const { sxBackground, children, backgroundOpacity } = props;
  const [chartType, setChartType] = React.useState('Thirsty');
  const [open, setOpen] = React.useState(false);
  const handleClose = () => {
    setOpen(false);
  };
  const handleOpen = () => {
    setOpen(true);
  };
  const handleChange = (event) => {
    setChartType(event.target.value);
  };
  const chooseVideo = (name) => {
    switch (name) {
      case "Thirsty": return Thirsty; break;
      case "hungry": return hungry; break;
      case "brother": return brother; break;
      case "Help": return help; break;
      case "no": return no; break;
      default: return Thirsty; break;
    }
  };

  return (
    <MainBannerLayoutRoot>
      <Typography variant="h4" marked="center" align="center" component="h2">
        PRACTICE
      </Typography>
      <Box sx={{ display: "flex", justifyContent: "center", paddingTop: "1em" }}>
        <video src={chooseVideo(chartType)} autoPlay loop muted style={{ objectFit: 'contain', width: '70%', height: '10%', display: "block", boxShadow: "rgba(149, 157, 165, 0.2) 0px 8px 24px" }}>
          Your browser does not support the video tag.
        </video></Box>
      {/* <Button onClick={handleOpen}>Show backdrop</Button>
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={open}
        onClick={handleClose}
      >
        <CircularProgress color="inherit" />
      </Backdrop> */}
      <Box sx={{ paddingTop: "1em", paddingBottom: "3em", display: "flex", justifyContent: "center" }}>
        <FormControl >
          <InputLabel id="demo-simple-select-label">ASL</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={chartType}
            label="Age"
            onChange={handleChange}
            sx={{backgroundColor:"#3ab09e",color:"white"}}
          >
            <MenuItem value={"Thirsty"}>Thirsty</MenuItem>
            <MenuItem value={"no"}>No </MenuItem>
            <MenuItem value={"hungry"}>Hungry </MenuItem>
            <MenuItem value={"brother"}>Brother  </MenuItem>
            <MenuItem value={"Help"}>Help  </MenuItem>
          </Select>
        </FormControl></Box>
    </MainBannerLayoutRoot>
  );
}

MainBannerLayout.propTypes = {
  children: PropTypes.node,
  sxBackground: PropTypes.oneOfType([
    PropTypes.arrayOf(
      PropTypes.oneOfType([PropTypes.func, PropTypes.object, PropTypes.bool]),
    ),
    PropTypes.func,
    PropTypes.object,
  ]),
  backgroundOpacity: PropTypes.number,
};

MainBannerLayout.defaultProps = {
  backgroundOpacity: 0.6, // Default opacity value of 0.5 (adjust as needed)
};

export default MainBannerLayout;
