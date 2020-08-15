#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:17:11 2019

@author: lin
"""

data = pd.read_csv(os.path.join(raw_data_path,train_path), index_col='family_id')
submission = pd.read_csv(os.path.join(raw_data_path,sample_submission_path), index_col='family_id')

