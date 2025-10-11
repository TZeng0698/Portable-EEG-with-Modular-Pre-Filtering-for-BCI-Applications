#include "ADS129X.h"

ADS129X ADS;

void setSamplerate(uint16_t rate) {
  uint8_t cfgVal = 0;

  // Map human-readable rate to ADS129x register values
  switch (rate) {
    case 16000: cfgVal = ADS129X_SAMPLERATE_16; break;
    case 8000:  cfgVal = ADS129X_SAMPLERATE_32;  break;
    case 4000:  cfgVal = ADS129X_SAMPLERATE_64;  break;
    case 2000:  cfgVal = ADS129X_SAMPLERATE_128;  break;
    case 1000:  cfgVal = ADS129X_SAMPLERATE_256;  break;
    case 500:   cfgVal = ADS129X_SAMPLERATE_512; break;
    case 250:   cfgVal = ADS129X_SAMPLERATE_1024; break;
    default:
      Serial.println("ERR: Invalid rate");
      return;
  }

  // Stop continuous mode before changing
  ADS.SDATAC(ADS_1);
  ADS.WREG(ADS_1, ADS129X_REG_CONFIG1, cfgVal);
  ADS.RDATAC(ADS_1);
  ADS.START(ADS_1);

  Serial.print("OK: Sample rate set to ");
  Serial.println(rate);
}

void setupLeadOffDetection() {
  // Configure LOFF register
  uint8_t regLOFF =
    (0b101 << ADS129X_BIT_COMP_TH2) | // Comparator threshold: 95% (default, see datasheet)
    (1 << ADS129X_BIT_VLEAD_OFF_EN) | // Enable lead-off detection
    (0b00 << ADS129X_BIT_ILEAD_OFF1) | // Current: 6 nA (00)
    (0b01 << ADS129X_BIT_FLEAD_OFF0); // Frequency: Do not use
  ADS.WREG(ADS_1, ADS129X_REG_LOFF, regLOFF);
  ADS.WREG(ADS_1, ADS129X_REG_LOFF_SENSP, 0x00); // Disable all positive channels
  ADS.WREG(ADS_1, ADS129X_REG_LOFF_SENSN, 0x00); // Disable all negative channels

}

void setup() {
  Serial.begin(115200); // always at 12Mbit/s
  Serial.println("Firmware v0.0.1");
  ADS.hw_reset();
  delay(3000);
  ADS.begin();
  Serial.print("ADS1298 ID = ");
  Serial.println(ADS.getDeviceId(ADS_1), HEX);

  uint8_t config1 =
    (0 << ADS129X_BIT_HR) |
    (0 << ADS129X_BIT_DAISY_EN) |
    (0 << ADS129X_BIT_CLK_EN) |
    (ADS129X_SAMPLERATE_1024); // Samplerate set to 250Hz
  ADS.WREG(ADS_1, ADS129X_REG_CONFIG1, config1);

  uint8_t config3 =
      (1 << ADS129X_BIT_PD_REFBUF) |
      (0 << ADS129X_BIT_VREF_4V)   |
      (1 << ADS129X_BIT_RLDREF_INT)|
      (1 << ADS129X_BIT_PD_RLD)    |
      (0 << ADS129X_BIT_RLD_LOFF_SENS) |
      (1 << ADS129X_BIT_RLD_MEAS);

  ADS.WREG(ADS_1, ADS129X_REG_CONFIG3, config3);

  ADS.WREG(ADS_1, ADS129X_REG_RLD_SENSN, (1 << ADS129X_BIT_CH1));
  ADS.WREG(ADS_1, ADS129X_REG_RLD_SENSP, (1 << ADS129X_BIT_CH1));
  
  setupLeadOffDetection();

  ADS.WREG(ADS_1, ADS129X_REG_CONFIG2, (1 << ADS129X_BIT_INT_TEST) | ADS129X_TEST_FREQ_2HZ);

  ADS.configChannel(ADS_1, 1, false, ADS129X_GAIN_1X, ADS129X_MUX_NORMAL);
  ADS.configChannel(ADS_1, 2, false, ADS129X_GAIN_1X, ADS129X_MUX_NORMAL);
  ADS.configChannel(ADS_1, 3, false, ADS129X_GAIN_1X, ADS129X_MUX_NORMAL);
  ADS.configChannel(ADS_1, 4, false, ADS129X_GAIN_1X, ADS129X_MUX_TEST);
  ADS.configChannel(ADS_1, 5, true, ADS129X_GAIN_1X, ADS129X_MUX_SHORT);
  ADS.configChannel(ADS_1, 6, true, ADS129X_GAIN_1X, ADS129X_MUX_SHORT);
  ADS.configChannel(ADS_1, 7, true, ADS129X_GAIN_1X, ADS129X_MUX_SHORT);
  ADS.configChannel(ADS_1, 8, true, ADS129X_GAIN_1X, ADS129X_MUX_SHORT);

  delay(1);
  ADS.RDATAC(ADS_1);
  ADS.START(ADS_1);
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.startsWith("SAMPLERATE")) {
    int rate = cmd.substring(10).toInt();
    if (rate > 0) {
      setSamplerate(rate);
    } else {
      Serial.println("ERR: Bad SAMPLERATE arg");
    }
  } else {
    Serial.println("ERR: Unknown command");
  }
}

void loop() {
  static String cmdBuffer;

  // --- Command handling ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdBuffer.length() > 0) {
        handleCommand(cmdBuffer);
        cmdBuffer = "";
      }
    } else {
      cmdBuffer += c;
    }
  }

  // --- Data streaming ---
  int32_t buffer[9];
  uint8_t packet[16];

  if (ADS.getData(ADS_1, buffer)) {
    packet[0] = 0xAA;
    packet[1] = 0x55;

    for (uint8_t ch = 1; ch <= 4; ch++) {
      int32_t val = buffer[ch];
      packet[2 + (ch - 1) * 3 + 0] = (val >> 16) & 0xFF;
      packet[2 + (ch - 1) * 3 + 1] = (val >> 8) & 0xFF;
      packet[2 + (ch - 1) * 3 + 2] = val & 0xFF;
    }

    packet[14] = 0x0D;
    packet[15] = 0x0A;

    Serial.write(packet, sizeof(packet));
  }
}
