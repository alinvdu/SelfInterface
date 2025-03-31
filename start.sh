#!/bin/bash
cd fast-api
gunicorn main:app -k uvicorn.workers.UvicornWorker --workers 1 --threads 16 --bind 0.0.0.0:$PORT
