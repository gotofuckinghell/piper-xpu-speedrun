# Где на XPU живёт обратный проход свёрток (oneDNN, а не «overrideable-фолбэк»)

Короткая запись, чтобы не повторять неверный вывод: имя функции
`aten::convolution_backward_overrideable` **не означает наивный обходной путь**.

## Что в исходниках Intel-сборки

```
aten/src/ATen/native/mkldnn/xpu/Conv.cpp        — XPU-свёртки, forward и backward
aten/src/ATen/native/mkldnn/xpu/detail/Deconv.cpp — transposed-ветка
third_party/ideep/mkl-dnn/src/gpu/intel/conv/jit/v2/planner — oneDNN jit-планировщик
```

`convolution_backward_overrideable` (Conv.cpp, ~строка 467):

- принимает градиент, вход и вес; допустимы 3D/4D/5D;
- типы: float, bfloat16, half, double — иначе `TORCH_CHECK` с понятным текстом;
- **1D приводится к 4D** через `view4d` (страйд/паддинг/дилатация пересчитываются);
- при `output_mask[0]` идёт `onednn::convolution_backward_data` (или
  `deconvolution_backward_data` для transposed);
- при `output_mask[1] || output_mask[2]` — `convolution_backward_weights` / deconv-вариант.

То есть это полноценная oneDNN-реализация dgrad/wgrad, а не im2col-заглушка.

## Почему это важно для оптимизации

1. Трейс, где `convolution_backward_overrideable` занимает ~25 % CPU, показывает **не отсутствие
   оптимизации**, а работу oneDNN. Своё Triton-ядро должно обогнать jit-планировщик oneDNN —
   это другой уровень задачи.
2. Прежде чем писать ядро, надо померить сам oneDNN dgrad на целевых формах
   (C_in, C_out, k=5, L=2048, stride 1 и 2, bf16) и сравнить с GEMM-эквивалентом.
   Если oneDNN упирается в память, Triton не поможет.
3. `view4d` бесплатен только для contiguous-тензоров; при нестандартных layout'ах возможны копии —
   проверять `.is_contiguous()` до вызова.

## Практический вывод

Оптимизировать стоит не «отсутствующее ядро», а:
- формат тензоров (contiguous, channels-last для 2D-ветвей дискриминатора);
- выбор stride/дилатации, при которых oneDNN берёт быстрый план;
- при подтверждённом профиците — своё Triton-ядро dgrad как GEMM (см. `triton-xpu-notes.md`).
