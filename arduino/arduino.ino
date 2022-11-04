#define USE_ARDUINO_INTERRUPTS true    
#include <PulseSensorPlayground.h>

int PulseWire = 0;
int Signal;       
int Threshold = 550;

PulseSensorPlayground pulseSensor;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);   

  pulseSensor.analogInput(PulseWire);   
  pulseSensor.setThreshold(Threshold);  


  if (pulseSensor.begin()) {
    Serial.println("We created a pulseSensor Object !");  //This prints one time at Arduino power-up,  or on Arduino reset.  
  }
}

void loop() {
  int myBPM = pulseSensor.getBeatsPerMinute();

  Serial.println(myBPM);

  delay(1000);
} 