Calvin Passmore

ECE 6600

2/2/23

# Homework 1

## Problem 1
How is a wireless network different from a wired network? Explain at least five differences.

---

1. A wireless network has no wires connecting all the devices

2. There is (generally) more noise in a wireless network

3. There could be wireless signal shadows when attempting to transmit a signal, whereas a wired signal doesn't have shadows as long as you are connected.

4. Wireless connections are more mobile

5. Wired connections don't need to worry about signal propagation, like the signal reflecting off of buildings. This causes wireless networks to have to worry about multipath propagation.

---
---

# Problem 2

Differentiate between a basic service set and an extended service set in IEEE 802.11.

---

A basic service set consists of one access point and must maintain a connection with the access point to keep a connection

An extended service set has multiple access points, and you only need to maintain a connection with one of them to keep a connection.

---
---

# Problem 3

What are the difference and relations between Wireless and Mobility?

---

Differences: Not all mobile devices are wireless, you could take a laptop with you to work in remote locations while not having a wireless network card in the laptop.

Similarities: Wireless networks allow for easier mobility, like cell phones, because you can maintain a connection while moving around.

---
---

# Problem 4

Please explain the pros and cons of using the analog repeater in wireless communications. 

---

Pros:
- You can amplify the signal so it can go longer distances
- A repeater is generally cheaper than a whole new station
- If placed correctly, it could cover signal shadows

Cons:
- You are introducing noise in the form of interference, which would lead to more errors in the data
- Cannot create a traffic separator, to reduce network noise
- Too many repeaters in a network can cause too much cross-talk

---
---

# Problem 5

Why are privacy and security more critical issues in the design of mobile wireless 
communication systems than in the design of conventional wired systems? 

---

In a wired network, the data is not available to anyone that is not connected with wires to the same network.

In a wireless network, anyone with a radio receiver can listen to wireless data. This opens a wireless network to much more vulnerability than a wired network.

---
---

# Problem 6

Study the works of Shannon and Nyquist on channel capacity.  Each places an upper bound 
limit on the bit rate of a channel. How are the two related?

---

They each describe the limit on how much data a channel can handle.

Shannon's equation describes a noisy channel while Nyquist describes a noiseless channel.

---
---

# Problem 7 

Given a channel with an intended capacity of 20 Mbps, the bandwidth of the channel is 3 MHz. 
What signal-to-noise ratio is needed to achieve this capacity?

---

C = B log<sub>2</sub>(1 + SNR)

20M = 3M log<sub>2</sub>(1 + SNR)

20/3 = log<sub>2</sub>(1 + SNR)

2<sup>20/3</sup> = 1 + SNR

SNR = 2<sup>20/3</sup> - 1

SNR &approx; 100.6

---
---

# Problem 8 

A digital signaling system is required to operate at 9600 bps. 

a. If a signal element encodes a 4-bit word, what is the minimum required bandwidth of 
the channel?

C = 2 * B * log<sub>2</sub>(2^4) = 9600

9600 = 2 * B * 4

B = 9600/8 = 1200 Hz

b. Repeat part (a) for the case of 8-bit words. 

---

C = 2 * B * log<sub>2</sub>(2^8) = 9600

9600 = 16B

B = 600 Hz

---
---
