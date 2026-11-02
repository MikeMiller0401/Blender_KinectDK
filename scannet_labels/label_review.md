# ScanNet200 labelling – choices to confirm

Every object in the apartment scene (364 instances) is labelled with a ScanNet200 class.
NYU40 and ScanNet20 maps are derived automatically from the official
`scannetv2-labels.combined.tsv`, so all three label sets stay consistent.
Changing any choice below only needs a re-run of post-processing (no re-rendering).

## No exact ScanNet200 class – labelled `object`
- Fruit in bowls (11 objects, dining table + kitchen island)
- Cutting board (kitchen)
- ~19 small metal bathroom fittings (taps, hooks, holders, small brackets)

## Judgement calls
| Scene object | Label used | Alternative |
|---|---|---|
| Fixed glass shower screen (bathroom) | shower door | shower wall |
| Reed-diffuser / incense sticks (~33 tiny pieces, bathroom) | decoration | object |
| Kitchen island body / top | kitchen cabinet / kitchen counter | – |
| Pendant lamps (dining, island) | ceiling light | lamp |
| LED cove strips, under-cabinet LEDs | light | – |
| Rugs and hallway runner | mat | – (no `rug` class) |
| Magazines | book | paper |
| Mirror frames; wardrobe mirror doors | mirror | wardrobe for the doors |
| Door casings / linings | doorframe | door |
| TV wall unit: low base / wall boxes | tv stand / shelf | cabinet |
| Glass jars on kitchen shelf | container | bottle |
| Tall built-in bathroom cabinet | bathroom cabinet | closet |

## Note on ScanNet20
Following the benchmark convention, classes outside the 20 (e.g. ceiling, mirror,
lamp, pillow, books, tv) are set to ignore (255) in the ScanNet20 maps.
