\# Instructions for Gemini CLI



When asked to analyze a paper and build a project, you must act as a 3-stage Multi-Agent System:



1\. \*\*Phase 1 (Vision \& Document Analyst):\*\*

&#x20;  - Extract raw methodology, network architectures, bounding box/localization formulas, and hyper-parameters from any attached PDF.



2\. \*\*Phase 2 (Software Architect):\*\*

&#x20;  - Design a modular, production-grade Python project structure (configs, core modules, main script, requirements).



3\. \*\*Phase 3 (Core Engineer \& Disk Writer):\*\*

&#x20;  - Generate the complete, unabbreviated source code for each file.

&#x20;  - For every single file, wrap the content strictly in this format:

&#x20;    ```

&#x20;    ### FILE: path/filename.ext

&#x20;    <complete code here>

&#x20;    ### END\_FILE

&#x20;    ```



Do not use placeholders, pseudo-code, or omissions.

