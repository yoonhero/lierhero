int PulseSensorPurplePin = 0;
int Signal;       
int Threshold = 550;

void setup() {
  Serial.begin(9600);   
}

void loop() {
  Signal = analogRead(PulseSensorPurplePin);

  Serial.println(Signal);

  delay(10);
} 