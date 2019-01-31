#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_losses tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_losses --with-doctest