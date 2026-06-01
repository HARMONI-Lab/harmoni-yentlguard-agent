#!/bin/bash
docker build -t test_app .
docker run --rm test_app ls -la yentlguard/yentlguard_ui/public
