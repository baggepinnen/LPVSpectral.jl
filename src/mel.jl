fft_frequencies(fs::Real, nfft::Int) = LinRange(0f0, fs / 2f0, (nfft >> 1) + 1)


"""
    MelSpectrogram{T, F, Ti} <: DSP.Periodograms.TFR{T}

# Arguments:
- `power::Matrix{T}`
- `mels::F`
- `time::Ti`
"""
struct MelSpectrogram{T, F,Ti} <: DSP.Periodograms.TFR{T}
    power::Matrix{T}
    mels::F
    time::Ti
end
DSP.freq(tfr::MelSpectrogram) = tfr.mels
Base.time(tfr::MelSpectrogram) = tfr.time

"""
    MFCC{T, F, Ti} <: DSP.Periodograms.TFR{T}

# Arguments:
- `mfcc::Matrix{T}`
- `number::F`
- `time::Ti`
"""
struct MFCC{T, F, Ti} <: DSP.Periodograms.TFR{T}
    mfcc::Matrix{T}
    number::F
    time::Ti
end
DSP.freq(tfr::MFCC) = tfr.number
Base.time(tfr::MFCC) = tfr.time


function hz_to_mel(frequencies)
    f_min = 0f0
    f_sp = 200f0 / 3

    mels = collect((frequencies .- f_min) ./ f_sp)
    min_log_hz = 1000f0
    min_log_mel = (min_log_hz .- f_min) ./ f_sp
    logstep = log(6.4f0) / 27f0

    @inbounds for i = 1:length(mels)
        if frequencies[i] >= min_log_hz
            mels[i] = min_log_mel + log(frequencies[i] / min_log_hz) / logstep
        end
    end

    mels
end


function mel_to_hz(mels)
    f_min = 0f0
    f_sp = 200f0 / 3
    frequencies = collect(f_min .+ f_sp .* mels)

    min_log_hz = 1000f0
    min_log_mel = (min_log_hz .- f_min) ./ f_sp
    logstep = log(6.4f0) / 27f0

    @inbounds for i = 1:length(frequencies)
        if mels[i] >= min_log_mel
            frequencies[i] = min_log_hz * exp(logstep * (mels[i] - min_log_mel))
        end
    end

    frequencies
end

function mel_frequencies(nmels::Int = 128, fmin::Real = 0.0f0, fmax::Real = 11025f0)
    min_mel = hz_to_mel(fmin)[1]
    max_mel = hz_to_mel(fmax)[1]

    mels = LinRange(min_mel, max_mel, nmels)
    mel_to_hz(mels)
end

"""
    M = mel(fs::Real, nfft::Int; nmels::Int = 128, fmin::Real = 0f0, fmax::Real = fs/2f0)

Returns a Mel matrix `M` such that `M*f` is a mel spectrogram if `f` is a vector of spectrogram powers, e.g.,
```julia
M*abs2.(rfft(sound))
```
"""
function mel(fs::Real, nfft::Int; nmels::Int = 128, fmin::Real = 0f0, fmax::Real = fs/2f0)
    weights = zeros(Float32, nmels, (nfft >> 1) + 1)
    fftfreqs = fft_frequencies(fs, nfft)
    melfreqs = mel_frequencies(nmels + 2, fmin, fmax)
    enorm = 2f0 ./ (melfreqs[3:end] - melfreqs[1:nmels])

    for i in 1:nmels
        lower = (fftfreqs .- melfreqs[i]) ./ (melfreqs[i+1] - melfreqs[i])
        upper = (melfreqs[i+2] .- fftfreqs) ./ (melfreqs[i+2] - melfreqs[i+1])

        weights[i, :] = max.(0, min.(lower, upper)) * enorm[i]
    end

    weights
end


"""
    melspectrogram(s, n=div(length(s), 8), args...; fs=1, nmels::Int=128, fmin::Real=0.0f0, fmax::Real=fs / 2.0f0, window=hanning, kwargs...)

DOCSTRING

#Arguments:
- `s`: signal
- `n`: number of points in each window
- `args`: are sent to `spectrogram`
- `fs`: sample frequency
- `nmels`: number of mel frequencies
- `fmin`: minimum freq
- `fmax`: maximum freq
- `window`: window function, defaults to hanning
- `kwargs`: are sent to `spectrogram`
"""
function melspectrogram(S::DSP.Periodograms.Spectrogram; fs=1, nmels::Int = 128, fmin::Real = 0f0, fmax::Real = fs / 2f0)
    n = 2size(S.power,1)-1
    data = mel(fs, n; nmels=nmels, fmin=fmin, fmax=fmax) * S.power
    nframes = size(data, 2)
    MelSpectrogram(data, LinRange(hz_to_mel(fmin)[1], hz_to_mel(fmax)[1], nmels), S.time)
end

function melspectrogram(s, n=div(length(s), 8), args...; fs=1, nmels::Int = 128, fmin::Real = 0f0, fmax::Real = fs / 2f0, window=hanning, kwargs...)
    S = DSP.spectrogram(s, n, args...; fs=fs, window=window, kwargs...)
    melspectrogram(S,fs=fs,nmels=nmels,fmin=fmin,fmax=fmax)
end


"""
    mfcc(s, args...; nmfcc::Int=20, nmels::Int=128, window=hanning, kwargs...)

Compute the [Mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

# Arguments:
- `s`: signal
- `args`: are sent to `spectrogram`
- `nmfcc`: number of coeffs
- `nmels`: number of mel frequencies
- `window`: window function, defaults to hanning
- `kwargs`: are sent to `spectrogram`
"""
function mfcc(s, args...; nmfcc::Int = 20, nmels::Int = 128, window=hanning, kwargs...)
    if nmfcc >= nmels
        error("number of mfcc components should be less than the number of mel frequency bins")
    end
    M = melspectrogram(s, args...; nmels=nmels, window=window, kwargs...)
    mfcc = dct_matrix(nmfcc, nmels) * power(M)

    for frame in 1:size(mfcc, 2)
        mfcc[:, frame] /= norm(mfcc[:, frame])
    end

    MFCC(mfcc, 1:nmfcc, time(M))
end

"""returns the DCT filters"""
function dct_matrix(nfilters::Int, ninput::Int)
    basis = Array{Float32}(undef, nfilters, ninput)
    samples = (1f0:2f0:2ninput) * π / 2ninput
    for i = 1:nfilters
        basis[i, :] = cos.(i * samples)
    end

    basis *= sqrt(2f0/ninput)
    basis
end
