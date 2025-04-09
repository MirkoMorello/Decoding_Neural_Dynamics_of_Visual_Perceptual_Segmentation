## Retina
- It does significant pre-processing
- **Retinal Ganglion Cells (RGCs)**  are the output neurons
- **Receptive field** each RGC responds to light only in a specific small region of the visual field. Many have a "center surround" structure, like being excited by light in the center and inhibited in the surround area, or eve vice versa, this makes them good at detecting edges and spots.
- The analagy to ML world is that the earliest layers of processing maybe akin to input normalization or basic filtering, but spatially localized.

## Lateral Geniculate Nucleus (LGN)
- A relay station in the thalamus. Receives inout from the RGC. It's often thought as just a relay, but it likely plays roles in attention and gating information flow. It mantains the spatial mapping (retinotopy) and segregation of information from the two eyes.
- The analogy to ML could be a bottleneck layer or attention mechanism gating input to the main processor.
## V1: the primary Visual Cortex (Striate Cortex) 
- This is where the _cortical processing_ of vision truly begins, it's arguably the most understood visual area. Located in the occipital lobe.
	![[Pasted image 20250409145546.png]]
- Retinotopy: V1 has a precise map of the visual field. Adjacent neurons respond to adjacent parts of the visual world.
- **Hubel & Wiesel's Nobel prize discoveries**:
	- Simple cells: Respond best to bars of light or dark edges of a specific orientation at a specific location in their receptive field. They have discint ON (excitatory) and OFF (inhibitory) subregions/ If a bar aligns perfectly with the ON region the cell fires strongly. Misalighn it, or put it in the OFF region, and it responds weakly or is inhibited.
	- They might be built by summing inputs from several LGN cells whose center-surround RFs are aligned in a row.
	- The ML anaolgy might be that these are like the oriented Gabor filters we see in the first layer of many CNNS trained on natural images. They are localized edge/bar detectors
	- ![[Pasted image 20250409160824.png]]
	- 