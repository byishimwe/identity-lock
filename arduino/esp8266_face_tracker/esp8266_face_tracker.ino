#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>

// ==========================================
// FaceLocking: ESP8266 Stepper Motor Controller
// Controls 28BYJ-48 Stepper via ULN2003 Driver
// ==========================================

// --- WiFi Credentials ---
const char* ssid = "Pessi";
const char* password = "michael@069";

// --- MQTT Broker Settings ---
// Your computer's IP address on the Wi-Fi network:
const char* mqtt_server = "192.168.1.151";
const int mqtt_port = 1883;
const char* mqtt_topic = "face_track";

WiFiClient espClient;
PubSubClient client(espClient);

// --- ULN2003 Pin Connections ---
// IN1 -> D1 (GPIO 5)
// IN2 -> D2 (GPIO 4)
// IN3 -> D5 (GPIO 14)
// IN4 -> D6 (GPIO 12)
// AccelStepper 4-wire sequence: IN1, IN3, IN2, IN4
#define IN1 5   // D1
#define IN2 4   // D2
#define IN3 14  // D5
#define IN4 12  // D6

AccelStepper stepper(AccelStepper::HALF4WIRE, IN1, IN3, IN2, IN4);

// Center position (home)
long targetStepPosition = 0;

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  message.trim();

  Serial.print("[MQTT] ");
  Serial.print(topic);
  Serial.print(" => ");
  Serial.println(message);

  // Command handlers:
  if (message == "LEFT") {
    // Relative step counter-clockwise
    stepper.move(-40);
  } else if (message == "RIGHT") {
    // Relative step clockwise
    stepper.move(40);
  } else if (message == "CENTER") {
    // Return to home center (0)
    stepper.moveTo(0);
  } else if (message == "STOP" || message == "NOT_DETECTED") {
    // Hold current position
    stepper.stop();
  } else {
    // Numeric target position for "tacho" pointer indicator
    bool isNegative = false;
    int startIdx = 0;
    if (message.startsWith("-")) {
      isNegative = true;
      startIdx = 1;
    } else if (message.startsWith("+")) {
      startIdx = 1;
    }

    long pos = message.substring(startIdx).toInt();
    if (isNegative) {
      pos = -pos;
    }

    // Clamp to safe range (+-800 steps, approx +-70 degrees)
    pos = constrain(pos, -800, 800);
    stepper.moveTo(pos);

    Serial.print(" -> parsed pos: ");
    Serial.print(pos);
    Serial.print(" | current: ");
    Serial.println(stepper.currentPosition());
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT broker at ");
    Serial.print(mqtt_server);
    Serial.print("... ");
    String clientId = "ESP8266-FaceLock-" + String(random(0xffff), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("CONNECTED!");
      client.subscribe(mqtt_topic);
      Serial.print("Subscribed to topic: ");
      Serial.println(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" (will retry in 2s)");
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n======================================");
  Serial.println("  FaceLocking ESP8266 Stepper Tracker ");
  Serial.println("======================================");

  // Configure Stepper: 600 maxSpeed gives much higher torque to prevent stalling in reverse
  stepper.setMaxSpeed(600.0);
  stepper.setAcceleration(300.0);
  stepper.setCurrentPosition(0); // Start position is Center (0)

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("ESP8266 IP Address: ");
  Serial.println(WiFi.localIP());

  // Configure MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  stepper.run(); // Keeps the stepper moving smoothly
}
