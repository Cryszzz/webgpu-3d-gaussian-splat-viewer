import { parse } from '@loaders.gl/core';
import { PLYLoader } from '@loaders.gl/ply';
import { Float16Array } from '@petamoriken/float16';
import { mat3, Quat, quat, Vec3, vec3 } from 'wgpu-matrix';
import { log, time, timeLog } from './utils/simple-console';
import { sigmoid } from './utils/util';
import { decodeHeader, readRawVertex ,nShCoeffs} from './utils/plyreader';

const c_size_float = 2;   // byte size of f16
// const c_size_float = 4;   // byte size of f32

const c_size_2d_splat = 
  4 * c_size_float  // rotation
  + 2 * c_size_float  // screen space position
  + 4 * c_size_float  // color (calculated by SH)
;

const c_size_3d_gaussian =
  3 * c_size_float  // x y z (position)
  + c_size_float    // opacity
  + 6 * c_size_float  // cov
;

function build_cov(rot: Quat, scale: Vec3): number[] {
  const r = mat3.fromQuat(rot);
  const s = mat3.identity();
  s[0] = scale[0];
  s[5] = scale[1];
  s[10] = scale[2];
  const l = mat3.mul(r, s);
  const m = mat3.mul(l, mat3.transpose(l));
  // wgpu mat3 has 4x3 elements
  // [m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]]
  return [m[0], m[1], m[2], m[5], m[6], m[10]];
}

export type PointCloud = Awaited<ReturnType<typeof load>>;

export async function load(file: File, device: GPUDevice) {
  const blob = new Blob([file], { type: file.type });
  const arrayBuffer = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = function(event) {
      resolve(event.target.result);  // Resolve the promise with the ArrayBuffer
    };

    reader.onerror = reject;  // Reject the promise in case of an error
    reader.readAsArrayBuffer(blob);
  });

  const [vertexCount, propertyTypes, vertexData] = decodeHeader(arrayBuffer);
  // figure out the SH degree from the number of coefficients
  var nRestCoeffs = 0;
  for (const propertyName in propertyTypes) {
      if (propertyName.startsWith('f_rest_')) {
          nRestCoeffs += 1;
      }
  }
  const nCoeffsPerColor = nRestCoeffs / 3;
  const sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
  const num_coefs = nShCoeffs(sh_deg);

  const c_size_sh_coef = 
    3 * num_coefs * c_size_float // 3 channels (RGB) x 16 coefs
  ;

  // figure out the order in which spherical harmonics should be read
  const shFeatureOrder = [];
  for (let rgb = 0; rgb < 3; ++rgb) {
      shFeatureOrder.push(`f_dc_${rgb}`);
  }
  for (let i = 0; i < nCoeffsPerColor; ++i) {
      for (let rgb = 0; rgb < 3; ++rgb) {
          shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
      }
  }

  const num_points = vertexCount;

  log(`num points: ${num_points}`);
  log(`processing loaded attributes...`);
  time();

  // xyz (position), opacity, cov (from rot and scale)
  const gaussian_3d_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * c_size_3d_gaussian,  // buffer size multiple of 4?
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  // Spherical harmonic function coeffs
  const sh_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * c_size_sh_coef,  // buffer size multiple of 4?
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const sh = new Float16Array(sh_buffer.getMappedRange());

  var readOffset = 0;
  var gaussianWriteOffset = 0;
  var positionWriteOffset = 0;
  for (let i = 0; i < vertexCount; i++) {
    const [newReadOffset, rawVertex] = readRawVertex(readOffset, vertexData, propertyTypes);
    readOffset = newReadOffset;

    const o = i * (c_size_3d_gaussian / c_size_float);
    const output_offset = i * num_coefs * 3;

    for (let order = 0; order < num_coefs; ++order) {
        const order_offset = order * 3;
        for (let j = 0; j < 3; ++j) {
            const coeffName = shFeatureOrder[order * 3 + j];
            sh[output_offset +order_offset+j]=rawVertex[coeffName];
        }
    }

    const rot = quat.create(rawVertex.rot_1, rawVertex.rot_2, rawVertex.rot_3,rawVertex.rot_0);
    const scale = vec3.create(Math.exp(rawVertex.scale_0), Math.exp(rawVertex.scale_1), Math.exp(rawVertex.scale_2));
    const cov = build_cov(rot, scale);

    gaussian[o + 0] = rawVertex.x;
    gaussian[o + 1] = rawVertex.y;
    gaussian[o + 2] = rawVertex.z;
    gaussian[o + 3] = sigmoid(sigmoid(rawVertex.opacity));
    gaussian.set(cov, o + 4);
  }

  gaussian_3d_buffer.unmap(); 
  sh_buffer.unmap();

  const splat_2d_buffer = device.createBuffer({
    label: '2d gaussians buffer',
    size: num_points * c_size_2d_splat,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
  });

  timeLog();
  console.log("return result!");
  return {
    num_points: num_points,
    gaussian_3d_buffer,
    sh_buffer,
    splat_2d_buffer,
  };
}
