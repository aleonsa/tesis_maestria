function i_q = adc_quantize(i, bits, range)
%ADC_QUANTIZE  Simulates ADC quantization for current measurement.
%
%   i_q = adc_quantize(i, bits, range)
%
%   i     : [Nx1] continuous current value(s) [A]
%   bits  : ADC resolution [bits]  (e.g. 12)
%   range : full-scale magnitude [A]  (input clipped to ±range before quantization)
%
%   LSB = 2*range / 2^bits
%   For 12-bit, ±10 A: LSB ≈ 4.88 mA

lsb = 2 * range / 2^bits;
i_q = round(i / lsb) * lsb;
i_q = max(-range, min(range, i_q));   % clamp to ADC input range

end
