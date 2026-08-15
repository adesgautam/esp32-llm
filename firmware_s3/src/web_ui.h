#ifndef WEB_UI_H
#define WEB_UI_H

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

void init_web_ui();
void handle_web_ui_client();
void send_token_to_web_clients(const char* token);
bool is_generating_for_web();
const char* get_web_prompt();
void clear_web_prompt();

#endif
