// Vertical face‑tracker Arduino sketch for ESP8266
// This is a copy of the horizontal sketch (esp8266_face_tracker.ino) but adds a second
// stepper motor that controls the neck pitch (up‑down). The horizontal motor continues
// to use the original pins (IN1‑IN4) and this vertical motor uses a separate set of GPIO
// pins defined below. Both motors subscribe to the same MQTT topic ("face_track").
// The payload is interpreted as a step target; the same value is applied to both
// steppers, allowing the Python script to publish either horizontal or vertical
// offsets depending on which script is running.

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>

// ---------- Wi‑Fi ----------
const char* ssid = "YOUR_SSID";          // TODO: replace
const char* password = "YOUR_PASSWORD"; // TODO: replace

// ---------- MQTT ----------
const char* mqtt_server = "127.0.0.1"; // broker IP
const int   mqtt_port   = 1883;
const char* mqtt_topic  = "face_track";

WiFiClient espClient;
PubSubClient client(espClient);

// ---------- Horizontal stepper (left‑right) ----------
#define H_IN1 D1  // GPIO5
#define H_IN2 D2  // GPIO4
#define H_IN3 D5  // GPIO14
#define H_IN4 D6  // GPIO12
AccelStepper hStepper(AccelStepper::HALF4WIRE, H_IN1, H_IN3, H_IN2, H_IN4);

// ---------- Vertical stepper (up‑down) ----------
// Choose pins that are free on your board and are not boot‑strap pins (GPIO0,2,15).
// The example uses D7 (GPIO13), D8 (GPIO15), D0 (GPIO16) and D3 (GPIO0). Adjust as needed.
#define V_IN1 D7   // GPIO13
#define V_IN2 D8   // GPIO15 (boot‑strap, works if not used for flashing)
#define V_IN3 D0   // GPIO16
#define V_IN4 D3   // GPIO0  (boot‑strap, avoid if you flash frequently)
AccelStepper vStepper(AccelStepper::HALF4WIRE, V_IN1, V_IN3, V_IN2, V_IN4);

void setupWifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();
  Serial.print("WiFi connected, IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP8266FaceTracker_" + String(random(0xffff), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      client.subscribe(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  // Payload is expected to be a signed integer (steps) or a direction string.
  // We parse both the same way as the horizontal sketch.
  String msg;
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }
  msg.trim();
  Serial.print("[MQTT] Received ");
  Serial.println(msg);

  // Try to parse as integer
  long target = 0;
  bool isNumber = true;
  if (msg.length() == 0) {
    isNumber = false;
  } else {
    // Handle optional leading '+' or '-'
    int startIdx = (msg[0] == '+' || msg[0] == '-') ? 1 : 0;
    for (int i = startIdx; i < msg.length(); i++) {
      if (!isDigit(msg[i])) {
        isNumber = false;
        break;
      }
    }
    if (isNumber) {
      target = msg.toInt();
    }
  }

  if (isNumber) {
    // Clamp to reasonable range for safety
    if (target > 800) target = 800;
    if (target < -800) target = -800;
    // Move both steppers to the same target (or you can separate logic here)
    hStepper.moveTo(target);
    vStepper.moveTo(target);
    Serial.print("[Stepper] Move to ");
    Serial.println(target);
  } else {
    // Direction strings – we only implement LEFT/RIGHT for horizontal and UP/DOWN for vertical.
    String cmd = msg;
    cmd.toUpperCase();
    if (cmd == "LEFT") {
      hStepper.move(-200);
    } else if (cmd == "RIGHT") {
      hStepper.move(200);
    } else if (cmd == "UP") {
      vStepper.move(-200); // negative moves up due to motor wiring; adjust if needed
    } else if (cmd == "DOWN") {
      vStepper.move(200);
    } else if (cmd == "CENTER" || cmd == "STOP" || cmd == "NOT_DETECTED") {
      // stop both
      hStepper.stop();
      vStepper.stop();
    }
    Serial.print("[Cmd] Executed ");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);
  delay(10);

  // Initialise steppers
  hStepper.setMaxSpeed(600.0);
  hStepper.setAcceleration(300.0);
  vStepper.setMaxSpeed(600.0);
  vStepper.setAcceleration(300.0);

  setupWifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  // Run steppers
  hStepper.run();
  vStepper.run();
}
