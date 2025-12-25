# Photon
Photon is a ecosystem that converts a grayscale video into a colorful video in real time. The reason why a grayscale or colorless video is captured is its nature to be captured in dark environment with no light sources using Infrared Cameras and Thermal Cameras. Infrared illumination is not as invasive as a high beam flash light that can cause irritation, disturbances and in some cases awareness to others which is addressed by Photon.

## Preview

<table>
  <tr>
    <td align="center">
      <p><strong>Captured</strong></p>
      <video width="320" autoplay loop muted playsinline>
        <source src="./captures/Captured.mp4" type="video/mp4">
      </video>
    </td>
    <td align="center">
      <p><strong>Processed</strong></p>
      <video width="320" autoplay loop muted playsinline>
        <source src="./captures/Processed.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

The processing of the grayscale video to make it colorful is done in real-time. The input resolution was limited to 256x256 pixels in order to reduce the latency and performance required to perform this alll computation in real time.

## [Hardware](./photon/hardware/)

The active illumination is done using Infrared LEDs which produce invisible infrared led that is captured by the Infrared Camera. There are two parts to properly powering such LEDs. Voltage regulator simply regulates a constant voltage and the current draw is dependent upon the load. Therefore to safely operate a diode which has exponential IV curve needs a current regulation given by following circuit.
<p align="center">
    <img src="./hardware/Schematic.png" alt="Image 1" width="90%"/>
</p>

This current limiter was soldered and realized to illuminate our video captures. Similarly even the Infrared camera we used is simply a pi camera whose IR filter was removed by us.

<p align="center">
    <img src="./hardware/photos/Photo 1.jpg" alt="Image 1" width="90%"/>
</p>



