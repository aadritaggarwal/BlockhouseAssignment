## Acknowledgements
This project was drafted and refined by Aadrit Aggarwal, with assistance from OpenAI's GPT‑based language models.

---

# Order‑Flow Imbalance (OFI) Feature Library

A compact Python toolkit that converts raw limit‑order‑book (LOB) event data into four market‑microstructure factors widely used for short‑horizon price forecasting:

* **Best‑Level OFI** – imbalance at the top of book (level 1)
* **Multi‑Level OFI** – per‑level imbalance across the first _N_ depth levels (default 10)
* **Integrated OFI** – single scalar feature obtained by projecting multi‑level OFI onto the first PCA component
* **Cross‑Asset OFI** – pressure on each asset coming from the OFI of its peers (partial sum **and** regression‑based variants)

The code follows the methodology of *Cross‑impact of order‑flow imbalance in equity markets* (Gomes & Waelbroeck 2023) and of Cont et al. 2014.

---

## Reference
> Gomes, J. & Waelbroeck, H. (2023). *Cross‑impact of order‑flow imbalance in equity markets.*  
> Cont, R. et al. (2014). *Price dynamics in a Markovian limit order market.*

---

