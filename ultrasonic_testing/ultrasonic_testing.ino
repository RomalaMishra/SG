#include <NewPing.h>
int buzzPin =(8); 
int trigPin = (10); 
int echoPin = (9); 
int duration, distance; 
void setup()
{
pinMode (buzzPin, OUTPUT); 
pinMode (trigPin, OUTPUT);
pinMode (echoPin, INPUT);
}
void loop()
{
digitalWrite (buzzPin, LOW); 
digitalWrite (trigPin, HIGH);
delay(50);
digitalWrite (trigPin, LOW);
duration=pulseIn(echoPin,HIGH);
distance=(duration/2)/29.1;
if(distance <=30) 
digitalWrite (buzzPin, HIGH);
delay(50);
if(distance >=30)
digitalWrite (buzzPin, LOW);
delay(50);
Serial.print("Distance 1: ");
Serial.println(distance);
}