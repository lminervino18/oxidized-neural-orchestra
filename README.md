# Final Degree Project – Distributed Training and Inference of Neural Networks in Rust

This repository contains the development of our Final Degree Project, focused on building a fully distributed, high-performance system for training and inference of neural networks using the Rust programming language.

## Overview

Modern machine learning systems must process massive datasets efficiently and reliably. To address the computational demands of training deep neural networks, this project proposes a distributed solution that scales horizontally across multiple interconnected nodes.

The system will be implemented entirely in Rust — a language well-suited for low-level systems programming due to its strong guarantees on memory safety, concurrency, and performance.

## Project Objectives

The main objective is to design and implement a modular and extensible infrastructure capable of training and evaluating neural networks in a distributed environment.

### Specific Goals

- **Neural Network Library**  
  Build a neural network library from scratch in Rust, without relying on external machine learning frameworks.

- **Distributed Architecture**  
  Design a distributed system with coordinated nodes capable of executing training and inference tasks in parallel.

- **Gradient Synchronization**  
  Implement at least one distributed gradient synchronization strategy, such as:
  - **All-Reduce**: a decentralized, synchronous method.
  - **Parameter Server**: a centralized, typically asynchronous approach.

- **Data Parallelism**  
  Enable parallel training over large datasets by partitioning data across worker nodes.

- **Distributed Inference**  
  Support batch predictions across the cluster with load balancing and minimal latency.

- **Performance Evaluation**  
  Benchmark the system under various scenarios, including heterogeneous hardware configurations and high data volume loads.

## Why Rust?

Rust offers an ideal balance between **performance**, **control over system resources**, and **safe concurrency**. These features make it an excellent candidate for developing low-level infrastructure where reliability and efficiency are essential.

This project not only explores distributed machine learning but also demonstrates how Rust can serve as a foundation for scalable AI systems.

## Team

- **Lorenzo Minervino** – 107863  
- **Marcos Bianchi** – 108921  
- **Alejo Ordoñez** – 108397  

UBA
Faculty of Engineering – Computer Engineering

## Bibliography

This repository includes a bibliography folder containing academic and technical resources reviewed throughout the project development process.

## License

This project may be released under an open-source license upon completion. Until then, all rights are reserved to the authors.
