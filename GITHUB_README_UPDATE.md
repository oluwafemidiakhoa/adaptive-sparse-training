# GITHUB README - ADD THIS SECTION

Add this after the "Key Results" section in your README:

---

## ⚠️ Scope and Limitations

### What This Validates

✅ **Core concept**: Adaptive sample selection maintains accuracy while using 10% of data
✅ **Controller stability**: PI control with EMA smoothing achieves stable 10% activation
✅ **Energy efficiency**: 89.6% reduction in samples processed

### What This Does NOT Claim

❌ **Not faster than optimized training**: My baseline is unoptimized SimpleCNN. For comparison, [airbench](https://github.com/KellerJordan/cifar10-airbench) achieves 94% accuracy in 2.6s on A100
❌ **Not SOTA on CIFAR-10**: This is proof-of-concept, not competition with state-of-the-art
❌ **Not production-ready**: Needs validation on larger datasets and modern architectures

### Honest Baseline Comparison

**My setup:**
- Model: SimpleCNN (3 conv layers)
- Hardware: Consumer GPU (GTX 1660 / similar)
- Training: Unoptimized, basic data augmentation
- Time: 120 min → 10.5 min with AST (11.5× speedup)
- Accuracy: ~60% → 61.2% with AST (same or slightly better)

**State-of-the-art (for reference):**
- [airbench](https://github.com/KellerJordan/cifar10-airbench): 94% in 2.6s on A100 (optimized everything)
- This work focuses on **sample selection efficiency**, not training optimization

### The Real Question

**Does adaptive selection add value ON TOP OF optimized training?**

That's the next validation step. Current work proves:
1. Concept works on unoptimized baseline ✓
2. Sample selection > random sampling ✓
3. PI controller maintains stability ✓

**Next steps:**
1. Test adaptive selection with airbench-style optimizations
2. Validate on ImageNet with ResNet/ViT
3. Compare to curriculum learning baselines

### Why Start Here?

CIFAR-10 with simple setup isolates the variable: does adaptive sample selection work?

Answer: Yes, but needs validation at scale.

---

## 🤝 Contributing

**Critical experiments needed:**
- [ ] Test on optimized baselines (airbench, etc.)
- [ ] ImageNet validation with modern architectures
- [ ] Comparison to curriculum learning
- [ ] Multi-GPU/distributed training
- [ ] Language model pretraining

If you're interested in collaborating on any of these, please open an issue or PR!

---
