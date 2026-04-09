#!/usr/bin/env python3
"""
Bug #0017: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['normalization', 'softmax_axis0_residual'], ['broadcast', 'mish_activation'], ['layout', 'resize_cubic_halfpixel'], ['fusion', 'add_mul_add_chain'], ['fusion', 'conv_bn_eval_explicit'], ['constant', 'unsqueeze_three_squeeze']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0017.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0017.onnx")
INPUT = np.frombuffer(bytes.fromhex("4aa85d3ea0e4bc3fa3d500bf1e5bc53e53769ebf8e30f13dbd112ebfca36e23fba3709bf32d5b1bcc506603f1c1f1a4094521ec0466fae3f9cdd433f4ebfb03f72b35a3f102d4abe7e0f9f3ff0c13e3faed2bcbf3245323e821a89bfb93abc3fc863db3e1579893e23ffa83f6911803fb1ea8bbf014d9bbf592e85bf3b43d23f7e3b1c3e331d69bf42873d3faa23943fa8ecac3ffbe2c33e38978fbf91f0913ff63f653e92ddbabf9e2b4a3f12fe3e3e711e363f73c9a7bfd2ce63bf88868a3e9d3bedbe73568c3ee1b3f5bd28d5c13f105c35bfdf2b02bfa27420c093bbb3bf7e295abff7220fbf7b3a823e3b1505bf0602e7bfc8f3ddbf90694abeca64abbfdf8a35bea57313406f076fbe4970543d11feb1bd7e0d4abf121295be47e10440665eba3fe873a93e44dba63f36f074bf7506303fd09aad3fb6dc3b3f5354a23f595d4f3e7b1cdebe07b0073f4e361f3fd7da9cbf0c1a32be43e3d6bffef5443f4dc6903e3cd11fbf39f6133ff02c503ffc1a5d3f4d249fbfd4395ebf5b289abf9978353f0d4d983f9b16a83fefa5debf75a4f4be204adc3fa7a3b3be7d86833f3274bebf43fc8dbf9629cc3e5c5130c08ef0474016b9da3e2678a43e723c053f1daee83fd07ab53d47a39ebe7ff6bd3e622d193f9125cabe0a6c2a3e716b8d3f4ae2453e7ee589bf6ee41cbf2d6d1a3fef532dbf72a1f23fbb1c09bfde3b16be9a5b07c057160fbf0e6c73bf9e376e3f749f953fe6ed04bfcbaa05bf1a14e33e9fa7a03f806454bf3226763f6efababf653c343fafdc3fbfb115623f237aa3be3cd9b03e70350cbf34f384bfaee1ab3f3e5f813f6e0089beeff283bf9557113ecc0ca33fe297963e5c9b9ebfdf4745bf1f2987bf5b401d3ec82724bebbfed9be7536fd3edaa7bc3ff0990c3e81e9a1bf6e819b3e72dbfabf579350bd3bc0493fd5c875bf8b86f13f275bf3bfd4fd5ebe715876bece35943f378cabbe4393b0bf0345babf8d5806bf38dd913faef5773fa36bdc3fd4c9593f7be0e13f8ef7ee3f4a6d4c3f18cf763efa50db3eb4b25a3e74c21cbf8fa2ae3fc6c0823fbdc29b3f9d379a3f02680ebf699da4bed05da13f60150fbfbb508abf475bbd3c1204d1bedc5032bdaa25ebbebc99e03fe63ac3bd7d18783fbc0d60bfaa732abf76596e3cfae2aebf0bd492bd19f4a0bf07b48c3f354fd13e6ead28bf4e338e3f97fd1f3e96babe3e2f4f57bfda46dd3f6e48c63f9e5db63f6f33ff3ea5f08cbdef1335bf1a5c36bfaed91f3ff8694bbfb8eab13f25055d3ed7b9d6be5433c8bdecb1be3ef078663f19ce163ff43b953eb1c3053e78e4afbf1d3b973f9df705be6c2a763f673b4cbf5d1581be49dc423f661922402a454840882ebe3e845ab3bc9ef7273f87de55bf93acca3ca208cb3ebce207c0c797b83fd5bf26bf8815ff3ff133323f5c358fbe1d1f0bbf8b5e4bbf2ff4d7be1b8396bf0df8863f54462c3f2988893f4c78be3ff7545cbf1ebf69bfeea7ebbf664ac53f2251e6be3264863faadd823e22cfe5be0b3dabbe3926a2bd713cd83e304905bf1b3ec2be3b677c3e232c483f8583a8bd23626fbf41ecfdbf95fa203fc9f1d23d869c8b3e10af1bbd65da6abf5d9d883f5ddf413f449cff3c959d46bf8790d6be467b4f3f2813b8bf688da5bf56e222bf0a8ed8bf7d38bf3d71523ebe09b9c5bd6ac69f3f4a10913fc104e3be7bc380bfd12d373f7888c03e02b695bf7cac70bfb538223e1c16553dc956d23e7e13543fd96d25bdb1cd563f3e152f3f650468bf7ed6a4beb2a89b3f7b3fdb3ed14634bfb2789bbf0ad5af3f7d67603f46bd04bf6d74afbf841ab9bf680d57bf2e8b66bec25503bfb36fe13e14d83240f53b06c0cef88c3f51b3903ee0e3e13db139c7bfee839ebf78d1983e3e42e9bf6bb8f0bea6709e3f5729793f1e15ecbe784ee53d38dbae3d4fb0ac3e46638bbcd71245bfc90bb03f0c0b12c08e8d433f253553bf4f7d8abdb4785fbf26e424406a638cbfed6857bf3c6d68be7fb7d5bf8e12103df03bc1bfc096753f04f3b23e1198273f86f24ebf983b46bd04fc01bf50b3b13fe21310c067c6bdbe9a89acbf226e813fd2a6ba3f3da41f3f30ee143f8a87aebeb29a823f5764eabe4e75e23eeba6093f4e53dbbe76e902c05cee19bfe40eed3d9d91b6be79a9d93e34516e3f1d7d95bfe0c4143f9ffc64bf0613253f78b6f63ee528a0be290424bf7724bd3e1580c23d8ff93dbfae94f6be550fe8be667316c0cc8b4b3f9359b0bfd3a27c3f0423953e14cf1f3f1f3439bf39311fbf221ac33f90db78beeaf05d3f3bc0cebe8f4ac03eadbaa53e8fab82beb2bcb33e22f88ebf18ad04c078d0e13e64ee53bf7418463f7acf2d3e4b79cf3ec837eb3ec1de1e3f842edb3e2e3ce23dbec55fbfc98795bf4644523f64332e3faff010be1283a93f97fc053d77aae73ffe1ce6bda2d79c3c59c6803fca928b3e8704d5be583cb73e15a08abf329f32bfc0f9eabed363993f2fe1b33d251a033f27dd08c012f3f63f1b3f27be3fa8b8bfd43de33d3372e13f9a76913d1b6190bf13712bbf5e74833d68eaeb3dcb5585be5c34204000c832bfe8f095bf46ad853fc594843f1b70a33e88a5e13f5e3662bfcadd80bf0130bebebde23d3ffb4e53bfd3c6e7bf1087b2bff08500bff523b23ed2eec7be9849ecbfa1417b3eeff1773f1dd1453f0e5dd0bf17337dbf56af4b3e563bddbfb84f08bf3fd15fbef53ce33eafff3d3ed0faccbf41b4613ffd9568bfd41534bf18c140bf836783bf6c0a9dbe290a97be51cd3a3f6a1e03c00639054025b3853e25f2343fd5d90bc02b9dec3f193ba4be6dbc40bec31e913ee0da943cffd5553f818b48be4e55ab3f0ac5cd3f22fc9bbeb175833f3feb17bfa4d838bf7d42cf3de229cb3d6feb1ac003fe11bf8fe590be97414fbf0858bf3d037891be635969bfb4c6973f6fb78bbe177ae23e2c4ef83e099c49be0b2233bf78bd713ca4a2be3fd24a1ec0dcf68c3f619f15be4f46933f825e1f40181bc83f18a0c23f13db7dbf7337b7be02a71a3f2358a5bf7ebc203fdd0718bfaf73eebd5f642a3f5bdbccbf7cc7e33f7125183f462e204092e5a73f412390bf4d5e4d3f764ee33f66fcf1bec474653f151503bf5833ff3dbd0f84bd098e383f79cdb53e79920c3f378e0fbfb11b243f60a0a4bc732193bff8bd153ffea872bf6dcfecbf3181853fcb496d3f069d0cbf0767d9bfe1e385bf0434cbbf380206409843acbf6475f3be8218c7bea7e4174098d89b3fd0c12fbe9ea50540770d783f84870a3fc9e651be20aceb3e83af0ec0f7e370be0acfb23eafc348bf6993423fb9fd60bfd44585bfd366483ed07d30be2a10aa3fb4ddb53f991c143f263a11c07b00023ec91e213ff35620bf87d87fbe6b6276bfdabae4bfcb7c9abe53f7873e9246d83fadc6afbf1bbbcbbe6cd81bc0d7b52e3e0ffd0dbe5d92efbea92d78bf54ef743fcff8d83fcd13c33e945773bdbf25593f60c3613d5a8c9cbf228dfd3e72e0513f57ac983e2fc31ebfa058a83e7cab9a3fe4158d3c6b58da3f8dc5a03f02d9c43f9e9f323f66e9943eff77f33eb6086f3f5377933e4a6421bf5e2ba13ef65cb8bd3f0847bfa9dc6b3f71b97fbf76e269bf381966be509c7dbed2c4383e0254dbbd8e99653e71d8c5bf15d7cd3e07f27e3ffa64c2bfe5f6ba3ddfe478bfab5269bf452300bfbcf62bc002c2113ef25f18bfb16849bf09a5013f886a573e3a7d58bf00d5593ecc7cbbbe6fb308bfb2cf4ebfc9f9b3bd461ec63ec8b119be28b4103f05cd003ff37f833f213c31bfb179203f5ef12340f089803e514da5bf2504273f9b2d14404b47c6bf6565164002ec15bf46fc1fbfe9c1b1bfbfa297bf8aab9dbdfbb4993e0b08963e4c26bebf5e0222bf4b9890bf9bf9a4bef34daf3e8893373ebe2f1e3eb8f8503d992bc03f8a50093f89cf3b3fc37a58be1dc7cb3db6e706bedd9ec33e4d5b643f66c6aabe5a05b2be2024d13ec19f48bedfdbc73f6fe92abf2a9da33f8ca4433f05cbc93eb9f5813febce153ee5c2d8bfe4cc8cbf0e3d96bf1d840e402439b13ec10c82bfd736c23f0ae0b6bf85a5d2bf22cb9f3f188c1cbf8e48723fe59f49be015601c0d34cdc3f937b88bead00ae3fc336dfbffccda73f8342d4bd3361babe3a00e83def6990bf096c77bf1e785bbfee50903f0a35df3ee49d513f97ec283fe35427bd7b6f60be5bba943f999aa2bf84a1d2bfc25da33e8e8fee3e8ac03f3fe010573f92e2eabf7b6bb53e6cece1be031b8cbf8ae3c1bcb2c709be798ad03ed1c2a1bf1c570c3e4c82f1bc4d9ce9bedfc6e13e21a2eebea41c12bfdc9d56be0463603f8c9318bee22b58bdc04f96bfd4bceebef8da663f2577403eadac0bbc16712abfd188b1bf0b9916be8141563f29fbf5bd5073b1bdf231833ea0a03d3fe0956abf52ccaebf097f503f9d26b93fc5a2f4beef6838bfad32fcbe9e1311bf300df23fea67fcbe7b7915bf31d6c7bfff3c29bf3e6aa93f5b5a1a3f58c3a1ba12a4fbbf41ef253f86b3cabf2630a43f4c694abfa39ea23e6782443f8a4494bff7d92f3ef69549bf2550ae3e32142e3ea395c1bf7ff481bf2d673f3e6c351cbd4d0e143f9bb094bfbc16f1bf83d1ac3e814c9abec945e4be09a709c03eaae43f2379a9bf8b68c13fbe3ca93f8343d73e7e310bbfae1af0bdd2b28d3f642f8f3e3dfea63fc6d4c13dde3b03bfed66e43e02a97e3ec0a78b3eb64a863d6bc14fbf7cf3753fd0eb28be56dc3c3f817d03403d98633e11ebcf3f3719a2bfa396ba3cbdd35a3dd95eccbd30ed873face2ebbf87049c3e4b12a1bf194135bfffcd94bf626f693d798fdcbd742864bfbf8c543f9ae61fbffb46b03d26691dc09ccba03f556e2f3ffc200cc068e1353f9891823fe210283fade72bc05ed59abfa9274540adf3ba3da02590bfbb18813f4e366a3eb418f53f0fab9d3e3b02adbef7b61b40776fcfbe69d92d3f147266bfffbcc53db8879b3cadaf5ebdc4c3a53f82c06a3ff417673d34da993ff9a778bf19463dbea2b28c3fbd86933e65a492beda459f3e2a1c883efa5856be4d3eebbef6c6c03c9fb6583f1e05593e22a8d9bcd88747408b48ef3f78965e3e601e35c0c52895bf9a86f5bfb0f93dbed54e103ef3d38d3f395cb93f0438a63f8d17a7bf6c31a5bbcbc8383f5af0bcbd6712e43fba65823f097b403f51509abf2652723e132da53fe12c35bebd50803f52ff45c0a5ef97bf5ea920bfd46b10bfbc32603f39f754bf3815c5be225ec43f81920c3f15ea0dbea6041dbff492d83f36e8473f31e1e1beba16d2bfd8cf833daa77b13e335e093e7c2330c0e0f942bf1292e53fba08dbbf9476304030efc1bfb69bf53f07406d3f09530c40b6ea3f3fba1e0c3e7897cebec8271fbe00b1b5bf86982d3ffac6e03f186499be630489bf3117b5bfdc114abf714bd2bd3857bf3d9a5a573f71f8fe3e68b317bf83420e3f4cc8503f4d77023fe3c0533f71cf8fbfbc9f963ef5dce13ff57b39bf8bf3a53f252b613fce3eaebe2d4ebebe7773ab3e22174a3fce30a13f12c70040446094bfd9e611bf032f003f5467303f0c0b883f950e603eb0582fbcb80f613ec1c0c0bed8acd73d7153173fb03983bf87d706be4ef4ad3e70acf1bc95791040749d60bf5b85afbff6c8b1beb5cfa4bfb8e36ebfd87c5e3eafa702c0d66d063f40322cbf1072cdbefbfbe4be0af92ebfd8ac443f03d223bfbefd3c3fa6775abf6e0e26bd3f1d353f41e38b3e6426833f7d378abffeb3b1be4c0f20c0b4c105c01589673fde05873f8a8a1bbea7d7c9be02635fbfb996643f4b61d0bb244b61be0475ff3fae438b3f6dd8733f87f882bfeba2febf8ae902c0873fcfbf86a6473ff50dde3e780d6fbe5c407a3fb106e9be8ee59cbef435933f2cf4fbbfd24880bf4af484bff18689bf4f028cbe256e8bbf1e18be3ff9a28dbe480854bfa5a0363e38ace13f8869e93f577726bed1eda9bf71ae4e3f6831bb3da0e8fcbf118c1440c9c8603fcab034bee828ecbbde4a2f3f6b293b3fd34223be8d80323f2cdbe1be96381bc092478c3f3a7aa63e2a22f8bbcdc20cbeb06db3bb8e401ec0709fdb3f0d9a0ec05b573c3fe33e39bf9d9fdabf928e193eac7ac03e55fbcd3f3460d1bfecb1fd3ec89596beacb0af3ff9ec663f9819843fdd80853e18c6a1bf698cbabed9bedabd1aa401bfe5ccbbbf5ab0e53db47b493f32b100c006e9eb3f5cfb09bfb7618abf7caccc3b296fbd3eef7cd9bedb09c23e119e1b3ff4d85c3fe6bafe3e5b4603bf3921733f5cc0473ffb0a9b3e5f00bdbffec8273ef3084bbf05fda73f5487563f9b5d65be5cabcbbf987ed2bf0d3d423fec82abbfb6f2f9be14be133fea6bacbfb6afb7bd65681ebfbf43713f04018d40403cfe3f428b9fbe150ade3e0ff35c3d4c57f73d8542ae3e26aa07c0c65c42bf0537a23f8eda8a3ec1a681bef2fbd6be7d98393f856f223f0b40babfe887103f1e6248bf5c0301bffc1cabbf5f951cbf63c90e3fbcd6ecbe2d77a23e0f27833e2e166a3eb95099be96185cbf34d9003fbcc03f3f68571abc22919cbff8b4633efa365b3fed7d3c3f4564143f992b8a3f436127bf6dd6853f5688e4bf38ba83bfede5523ea40b2b3fc9670bbed6580fbf382a15beed79763f9771debf581cbcbfd25d803e8595fa3d5fdd8b3fb75f9bbe4ca8e83e4b5d3cbfd10e683f437e433f662d04bf0aed5dbf19dcce3dea9de63fca826cbecde11b3f15cee1be1aa3a4be1ee08a3f408cf23ef02288bfe26d36bfcfb4c73f7a7af7be553d483fc80fefbf0bf29b3fe629f23e40b8113ea2638abf36256cbd7ba694bf848d11c05e2fa0bf8f7c283f742b11be6ed7c0be8759143e33add83e545decbdee98c43eb7efb9be438bb9be102a793f666969bf74a107be456b033ecff31ebf7e46833f5bc6113f59411bbe8e3290bd428c94bf3a34a1bf56e0e8be8628e03f583e83bf6b54f63f475b4bbfd1315fbf6f01dcbe1c6f2ebe30db27bde2c4063f02e1d0bdac9669bd96bd094065cf36bf3ad810bcfed4c43f33d98a3f1cd326bfe7989dbfec1d103f0e4351bf9ed5e3bef5a8b9bf2663f33e93b65dbfa5a68cbfea9d683e0234f5befce54ebf686e8d3ff279b3bfc957d6bfb661c1bf5bbc94bfd835f43f3b9b4dbf312ec73f8c79f23e896940bf7a55a7be162e99bc0b7d13bfc4689dbc8ba3abbe385dc6bf33f3903ffc883b402b893abe292c10403832b0bf87413bbc910d6ebe7dd354bfc177d93e2c48e4beb7f3d6be91d167be0aa4bd3d4c6434be4f50acbf6474c43f4dc9383f6aa0513e22c33cbfc97f1fbfe88a68bf1b4066bda68f2a3d283d9c3f3f73a3bfaa18b93f269333beae94d9bf675d96bf6d7fca3fc65c053e77550fc0199d32bf791f973f3646e43e1c1ea3bf740392bfef0c573e3f2ce8be9494403f023f963eb6042fbea1ed62bf187f31bfc8cc67bfb28137bfe4ba773f59b99c3f82409bbec979bdbf4d57fc3ddef97bbf11f235c096c2a03dd152b6bf3a42403f2648cebf03d3e53d38b080bf34407dbfb8301abf031c543ee908643f5b945c3e44c4833fbe503b3b8a1976be2f3f50bfceb2ce3faf56073f571c88bc831b38c06657de3f56d90c3f3170923feb543a3f64ffb2be4132bbbf0a77b83dafb53fbed58d7abf85b02dbe3602b13fe0b9953ec4309a3f52fc523e526d2e3fe9c8923d821e28bd403337c03b80793dd096efbe03878d3fefc78c3e514cd93e2dd581bfbc56533fa41a9bbf5b67d6be08978bbf17cf733f75ec323cdcb8f73fa58171bfa0979c3f99d6b7bef255c33ede93a5bf3f47623f34f51c3e0b1a1abf3219b93fbe48e1bd474f98bdd4bdefbe0d6ca83ec1ead5bf2e21b73e6cbc853f73109b3f87a42ebf261c13bee763c3bfacc773beace802bf5c23a33e8228913d1f381fbf7a5b093f7e03613d6f6f9cbe59c29abd91288fbf6e69a1bf506e813e36b3f53e9c11ae3e0ad692bf055f99bed5ad1440a0fe02bfa36a4e3f2e66ae3e5410453f53220b401e42743fcc37013fa84dbdbfaf4ac33f3b94923e89490bc0426f893f6d45fd3f3928573f92d7ae3e9b0712be302edbbf70dd07bf749f63bd48a32dbf3a0fc1bf2839fe3f2306e8bfc772693fd63a313ff40701c06fd20a3fdeeeadbf8fc0a03fd6c5afbefd7b46bed703213f57df2ebd4d1c3dbf6d9305c02a0c243f220a143c60a283bfbfd283bff2602dbfd87e0f3ff6adafbe43f17dbf7f7790bf6fa6a7bfdde44e3ee573fbbd6b1810bf25f1093fb65c8dbeea9c413ebe3bfd3e0086eeba8055b3bd1626ff3ffd73523ee908833eca8e613d1790103f60056ebd003029c081e5523f37e3b03e0fbcfe3e772b95bf18f26d3f9f741e3f86021b3e8b27c13ff726cabf756a8f3e09c8c03f9089e93d737701bf18c1963e8da2ba3fd9364ebf2396593f5f8485bfb338fdbfa99b1e40883ba0bef4e4a7bfdf1f263cd23cb93f5eef97be7cbec83e305d51bde9866e3f4f5312bf88d4253e932fc93f63d99b3eef25613e4e298f3d4ffcebbf20e7bb3ec9a821bfb1981e3e530a9fbd06f9103f86eed03e314bfbbebc46b2bf9b483bbf424b86bf4f6901bf8bf479bf5ca7a9bf42117dbf1212e0be0e92393e4d9b5fbfbfe7a83f8f62ce3fac1a2ebda0169abffc99a23fe87a9c3e5a2702409c85283f3d0df03f5f19af3ee89e803e7d318b3f4335014033b2c0be4262a53f90fe59bf476e0ebe0447ac3ee7d8cbbeee478c3f68a919c0550380bdd71703c0386b473e087425be827d17bfcdeda33e7f5a353df179653fea6acf3dfda286bd3d86bcbfff61b3be009863bf1d6b8e3f686ead3e3f5fbcbfab63933e2ea3213f75810cbed3f8cbbf637873bf26efe1be5478f8be1429063e7281cf3f078f07bf38c823bfb95c68bf068930bf6503a0bf2b6309bfd7fdf93f4d8091bfa07644bdd9256f3ff0f0c6bfbf6966be4a9bfe3eff510bbfc42f553fd0beb33dabc392bf5ee3c8bf39f645beb45bbfbf0c58d63f06cdac3e2d4ef53eb85ed23e0dbe243ea17edf3e6380843ffa1e223e9c21383f1c2a8d3e66b6d2be9d52e2bf315bd83f975f293f17508a3fdaf14e3fa11783bf34ca453e20d1e03ec01075bf110b873fd627b9bf2e4f633d8a1599bf436c4cbf8893a43f24edaebff1266abe151142bffe8f7b3eb646da3fd657a83fa9ce3ac0004580b97e3e623f907dae3ea8fe213fb510343f769c233f2ab738bf9b22a43fa989be3fa28f983eec5e9abe5dd72abf8e03183fb5dd823e8f2b84be3912d33daa20223fd2f163be4bb8933f03dc2a40c3b647bf0e95c0bdef1519c097255c3fa27ec0be48fc733f9f4ca5bfd8af923f3fe603be37d99a3f766ca43e4c52513f3188f63e041a26bed2ef1dbd523e4dbfe2fe4fbf5d9c8cbe6487a1becd0e3c402935773eeb7cdabf4c5e5fbffeb760bfa5735bbbb906ddbf6679b63fc7f7b63f82430abf0cedf3be67dc61bfd53caa3ed862483e00c323400f7acabe2ecbb13f61872ebf23b963be915d01bf041befbc38b412bfc9ddfcbee6360f3f96890bc0ea766a3e9774c3bf44a7c2be26b2a1bd12966ebd42928bbf41ba29bf2d3ae73e131905c03aa38c3f10c6da3f8ec729bf297f2d3f8e6a9a3f7c1ea33dc75d033f19f30a3f75720bbd1e642e3f88724d3f932fd7bfc70d853f98a747bf8cb8e4be0fd8d03fb462cf3dd092183ff6efa03f25e6903f47fd903f282c09bf5f5957bfb1461cbffae9283c89b0efbe05e9ed3f01d782bf7d20ecbf9e42943fae46c1bf04be66bf392f8f3e1b85013f826f95be21356c3f31be0b3f0bb6d8bb2176f03e59f6ddbf8c76253f01a69f3f9af0a43d75e2cebf701cc4bfc6a1d2bc7638e1bf380a0cbfa11c0b40aff962bf304c88bfb3df343e5caaa03fde8628bfa76fa23f22c7ec3f6ea962bfe4b0c0bf5d7767be2af84dbfbf95b03ff57adebf2525f93f0b82943fbd1f13bff61ed3bc4c27d33b0c7f35be87709e3da170863f5466023f1032483f75732d40dfa045bfab3c3cc0c474e43e1270793fc9a5aebd73e09bbf30ea7b3f87564ac08a6cc5be89298e3e4297883d55840cbf447382bf112c7ebf0cb9303fe51153bf0e8105401200bd3daa7a93bff2c48fbf994f7abfda0a43bf6fa046bfd5a0d83fdc06173f196077bda36e8c3f4b9ebc3dd47ec1beaf06c53e4e0d0240674118be5f9d1b405de22dbfa977903ede88933f1bef95bf7cf9a7be2bea4a3f3e56a43e923920bf4f72993d67106a3eb86dd43fa63ad2be8369b0bf1ca7cfbfb2944ebf75656ebfcbcf34bf8b15853f4848483f9baa54bf25009ebfdc8ed83f7822ab3c63fe7ebf04fe933ec73f083f47bea4bea741c63f2669e5bef108a73d4994703f3cc480bff73ed93e55c4443e1a7a40beafdf563efc4d943fe8780d407a75ffbf89254a3f6fea8b3d87402d40b602d63e78a309405240a43ff5315cbf10ebcabe1f1c023e6a296cbf5da2d9bf836f31bf3fb10bbf005966bf06894e3d8cc126403760a8bd8c2c8a3f90e5f23ecd62f2bf78a426bf8849013f26e216bf662ae1bf7dbd11bf9a180a403de2b0bf749daa3fa72e223f2f2680be673c353f588ecdbfe9789d3e87362bbe6ed65a3f57efc9becd6007bfbc25b53faa9592be262866bfc827faba6ffd4b3c1d0c4b3e188af7bec9e3a9bf1050b43f31ec6a3f4db55a3c1a832fbb4085b1bd3e50a33e0ca11ebd98aa56be4471853f5dd5f5bdec5e39be49333abe0c4f7e3ed34152bfce9c11bf96baea3fc8219f3f9495b7be496a5fbd90ea603fcf4312bfb43facbf4cf58e3f80a22d3c150bb3bce4b4823e510ad5bfeac6f1bfb3a609bf3643dcbe63621b3f109910bf2727a3bfebcc4fbe4727123fca146b3f5cd84fbffb79b63fe4b4f9bf24512abf482e983f1fe37e3fbf7c44bfc86e903ffac1013ff8862d3e9ad8893e98085d3cf03c8cbe999c7bbf4eb826bf981e873f9aa4993e2b4cdfbefb413dbf9d297f3eb7a62b3e524d253f18b9fabea4f7ff3f900d59bd9401323f4e2f6b3f6650f2bef27d9a3d1af02fbf9630583f6da8713fd4b6173fc782df3e642c28bfa24c5fbe66e036bd14d4e2bde06d77bde8bd2a4084af323f16f0df3f68c8053f5408803fd55dd9bffa27da3ec60da6be6472c43fecf0e8beb7d8983fce2111bf218bcdbfb607dfbf0463cf3d9eb1b83f3ce1043fc61b28be7c98213e142c6cbf14bb073e63fd693f39bdf7bc1bcfd5bf1536a93f0b6493bf35d6b33fa6a2513f190304be950b913ea9314f3fc6a4e2bf252a46c0b56c463f8b24d8bfc5e67dbd4c19f53ef65ea03ff9ebd0bf5cc98ebfe8f4c83e8076643f66a784bf442e3dbfda5937bebeeced3fb5e184bff262613ec40ca13ec7f6c23ea37d6e3ee6d300c012e88fbf1652bbbe3b941f3d15edbebfc94f603daa30eebe4fabbcbd87adae3f4d431d3fa11bee3f4369043f0b0fadbde0a0c7bed1be25bfb41201c03703123f5a96553e89d8f03f5ebb3240559b33bff1b513bec47fa73b180911bf3e3e493f5c66c83fb4c689bf9b4b51bf0f17cd3e6bbab73fa4b8febfd49addbfc4d7223f62746dbd97530e4095b1923e2349823fab71ce3ef8c36cbe1a82c43f99ef1dbf2e2302bf58e7913febed2bbeef6ac3bf1c778ebe471f3abfc9b98fbf09d933bfa054d73964845abff4bee53f7d25f7bfb7b0263f406da83e983abcbe7661683f3cec043fcd8d3ebfd3e3f73fd1fb1dbfe39e033ff071de3e52e7453eae23d9bf1e7fb03f4cadacbfd322153f3de87fbf0bf6f1bfa0f8adbf61835c3fdad28fbe5f296ebf3d36a1bf7f6e143f56e1babe1df1573fdab7ddbdb7b5163f6a32933ebd41513f5ee89d3e0b65f0beda68b3be7edd763f353e4c3fab08d23f5a3f983ec78a703fc5a9bd3ed31783bfd92ba6be86e0ec3f444d983f3ace2b3fb9b3463f62f64c3f34548bbe43e8f13f9e14f53ebebe853d8637b6bf024c9fbe2a4fe23e08e739bfea732ebf2ebf053ddcfee23f5ac5433e2ba0fc3f9affc23e4f1a453f5eafe4be305dd63f261615beaa2cd9bf5ea9393f699cdcbe2bc9df3fcd6097be488c063e9ca0893c0444ef3d8e76373ef9b0083f88b6d33e29ad8dbf98dc013f0de3b53e151aaf3f6296fb3e181f133f22c8893e73f54abfc157f53f1ab94fbe6027f6bffb1a0abf08a1ccbfa0cfb33f9ea5003ffb4063bff392bb3f6cb6e0bfc6118e3fd3938dbe4a9ab8bf33cbaabe30020b3f1a454cbe2c11323fe7f2dbbea69db3bfcb8e92bec4d6d1bf730fecbfe17aebbc2cb5c1bf3a1b973ee89665bfa45742bfdcc49fbec8ade7bf885672bf612c2fbff224cb3e452678bfaa6728bf218082bfe74a8b3fa674d4bfa2f0ec3e348ceb3ebf8645be788bffbe1087893fbf28a03eb06012c066c231bfdc7514bfed24c0bf1031633f09041c3ef5a10b4039edd03eaf72ce3eb21f07bf3c77c3beee470d3f7ffd6c3f5dfa6cbee4c18e3f07c123bf39e5993e41d0f83fdf8339bf6377383fb148ab3f920113bfefc08cbfa0e7b93bcb76aabf4824173fc26d923f81b8bc3f1ba7ae3fa003edbd9b168ebf6a8527401cc654bfdadc1b3ee90487bfe705eb3f2fd7293fd12a643e2144ee3fb8cd0c3f6f33b6be0487243f48d3bb3f68b20cc0d84016bde1c9a83f086e0340c6cd183f193fb53ea7af64bfe780bebffa6f13bfce300abf4f69383f1e50adbfd7eda2bfc968aebef1a58b3f982190bf78c3623e6f193b3f10a790bea8675cbfa2daca3ed2f0473f57ef963fa100af3fa9fffd3e27a603409383c43e29a6cd3e2ab0c5bf67aca0bfcabe6bbd9420d9beb5fac03fb1e132bf45520ebf8aba07c0507f253f90c0a83f739969bfb28daa3e45a785bf09aba6bf2a68edbe37bbc9bf2f4c3abfbba80440a2280bbf1ddfd33fc0dea43edf3c633f1825053fe82d223f8be8c23e278527bdce5c10be36480e3fcc65f73ed08e813fdf11603f87dde93fac85103f65447c3eca79943e97fa1dbf0bc1113fed8d86bed0a057bf0f3570bfc2f6d2bd056df73ee7af923f2539013fdf75af3cdd2c85bf6dda8cbf3fc844bf154df13fc06f113f46be8bbe536c4cbef7db65bf126de7bf679c03bef57d0ebf1292653e58ed78bf137c23bf1225e63c048786bf294ad0befd06323f155dd9bdcd1acc3e4459913e30c72bbe0efe573f467182bf44f944bfab24283f4e683e3d8c6210bfed97febe3e5d06c0bd2e1b403d5fab3e0a3e88bdb62ea8bfee43c13e7b8d92be5782c53fd94d983f6fecf93e2cf389be7b4c09c0a8b1ba3e9bfb1e3e8d7c803e8e0881bed64644bf3011d83f2a08a7beb34e8c3e3b70463fb0f32abe28f86dbcad97133e2cc3cb3ef2a53ebe2628a4be7039843f20aad23e1d5c26bfdc8b37bfbfe88abf3150a0bda57f833e35df913fcc6f8dbea74b80bfa5756abeb7d30140441557bf61fb2f3e0fd5483d646fb13e3432463ec2ba61c03a33373f0c7b833e76e953bfd7f702bd28f6fc3fe732e63f1eaa29bf97181a3baef9a33ffcfbc6be8839d03d6b8b8c3f18d148bf27eb7ebf2f66983e349c4c3f119ad6bebde66e3f832f233f013bc53f3eedeb3e8175973e2b486dbfc565993e0f4f873f0570393e3447373f6eeb203f00df07bfa76bd03f3c9fb83f972566be261d2d4063f5e4bff5c7723eb65643bf22c405bfc8213b3fad0dfc3ed35128be3e434cbde55501bebf9856be19ee013f7e4dea3f493b4ebf3f67023fff71c5bfab90f13e7de0d0bfb923c93dd364803f90dfac3e49232abe5b190140e223eabedaf776be9b8bbdbd84ee17be4aaa7b3f3049463ec4a3ca3edf41fe3f0dc1943e79ad113f17f0b33e5f9b923ff29007c0e863c5bba48c743d8b5cdfbea33d8ebe07a1943f4b14a73f0da181bf9bb6a53fea17ef3e1746263f50071dc082d1ea3e28a871bfdc779e3f6dbb4d3f80bd303fc31c96bf5817423e48cd6bbf2e445f3f535595bfc12529bfd91cf3be48e570bde4a9323ffbeecfbfefb021bfc5cca53e65cf20bec555613fc8e1443ffae496bf78d7a7bfb431a9bda3c609bffa1bac3f80ad86bfd8577fbfbeb5823dd280ac3fdacec8be2902193fe0ea673fd3db823f9f69ddbe1b7bb4bf0d32b4bfe5e7933f53f6823f11e7a2bfc2c1cebfdb891140536f6dbf99f4b1bd418543bf93ee583fb9e86c3f41e024bff4fc453f29320b3f594ea53e6fac93bec0ef923fd4478ebe1097933d46da5bbea03e76bf854388bdaae2a3bf35e9b03ff43d2abf00a29b3e0f309bbfe6ef53bf38d9a6bee154a5bfd1514bbffbd32cbe446b20bf405591be32284fbfad9287bf5633c33cbb53853fd3af8b3ff8a735bf43d67b3f1415d93f5740bdbe5e96aa3d4593863f9b4e90bf318cda3fce0da8bebee99cbe9a7efb3e8405edbe745b21bfd07ee03fe14f3abfd14dc7bfbef128bfeb0af83e8ea181bf320da3bf070724bf160f34be0646f83da0a69c3f6be113c0cdc71dbebd1d67bfa9fe1440922428bf134b1abd82d399bf6e19f8bfa3d15fbe8d00b73e1d71cf3f27740d407117413fe51fcc3fa2b6603f2420e5bfb44f5bbf6c7c943d394614bfd93b05401afb87bf37aba4be944c56bebd16113ce8d83dbf4ee78b3ca27559bc6dd18dbfa33e223f70caabbe4c3505406262313fb6c7ce3e2c448e3f872f82bf985e0bbfa46faabfcef98cbdaa82d1bdc5f48e3fb952f4bf2a06e8bf25540b3f61eeaa3f6f78303f54d065be4a3d57bf9822233f2391db3f6c1403bd80895cbeb39b2ebff759e13f596bcd3e1131093f0341dabf03dec23ed858ddbe91c275bf3892a9bf744e5b3f6d8ca1bf98b59fbf0f1745bfaae7d83f48e683bf2b6fb43f7d9fa1bf9fa8ed3c9a9985bfdfbe8d3e97db633ed2985ebfb6f24b3f1fba9f3dd526f43fcad304400b41aebfd296893f16ed9abe0d2613bf1e6300bfe8c9b03fa8d36c3f6a1af6bf09fde03ea6b0df3fb9f805c0a04337bfb229003e0fcf4cbfa2a09bbff4e3803eac35243f83590a3f2e3b4c3fe9ba73be6fca16bef62d71bece5c9d3eac14fbbe3f0a783f1ba5d0bfc5a0be3f3d3dbbbf1c8ca83fff8d1dbf175d94bf91cd2e3ff566e13f9fc0b63e9228ebbf51320740ff9d923f7366633f5759d83e4b8254bfbd9b7d3fde3c0440a8be823f7226c03fc9ae093f190586bef89b453ff00343bce7b562bfbf7b9c3f269b2cbd06a891bfc2dcb23e59a0be3fa738033fbc6e72bf7a61183d819342bf4ac13dbec755da3e4cbb033df46fcc3e29df3e3f57d1973f026228bfc42b31bd186c363e466b833f0f958e3fd40e83be506f133fd7f67d3d1247433f977c4ec05b60d4becfda24bf4e9b6d3f0e37adbebe49fdbf7363b43f09dcccbe3bdf7cbf41c8efbfcc4b75bf7e13ea3eb53fa73f4af0dfbfa4e9c73e9572043e2e02803de50a823f78b4013e6f9005c00fd588bf8e494dbf310c0f40365b5fbf7d84273eef2fdbbe978606bd9f3f61bf36ec43be52ca393e1e0de33e6578ea3fcadd8ebf565ab53f4a6f8bbf16c52e3f8106a93efa90353e2c02893e52b8f93e47a4d5bf1cf31c3f47db643f2653ad3ec0d3663ec520eabf915b953f6ee4a7bed1ea3abfd235a93fadac913ef2b99fbfe1200b3fccf53f3ea87867be941bb53f4d2aecbee6f15abfb9e672bf83bbfebeb9e582bfe8cb133fbbdec03f562e90bf1c7d2f3fe71bbf3d4250b7bf568c57bfe859c7be4a65a0be21144b3ebac2493fa848ccbb6b6045bfca0c613bc2b5bc3f4c18bbbe5a6132bf4213103f28c268bd7b41a03ea6ac5e3f43a3723e2f43cabfab1502bf8fe122bee6c3bd3fe0b3563fd8e041bfd55005bf2fd420c097e63bbfb644343ebb4fbb3f4cdecabe12932fbf4e4b993efa9f0e3ffd2b953d87a02dbf8040bebeaa6fd43d4a9585be94d3df3f968b76be2fd9af3fdc498cbf94dcb2bf286a96bf2a42693ebffc2cbf7ae89dbfa445473e4c24943fe69510bf23713e3fd878153f240e2f3f9d9dfd3f4071edbed25cabbe6e95843f8355bebf8c1221bf900cc6be821c903f3065d2bed8c3073e1daf453f4fbcbe3fdfa908c0d5967fbf079631bdb2a39dbee81876be40450dbf5aab85bd27df0140cf5c623f811104bfd69cfebfc0eeb73efb4a7ebf35ea9fbf100f59bf204b0fbf5f92033fda91393fd01550bd07234d3ee2f6353f5ab8fd3e027cb3bed004a2bf57d4e9be12cdbd3edc12f5bf0f9ecfbffe6be5bf15f96f3f738202c0d604ab3e2326dabffad4c0bfcd6f523f92fbb93ecd59793fba5f29bfe3091240dbd3703fd4e0eebe4c73f8bfff1f97bc413f423f7966a8bed7b1a73d27931240720c2c3f5b80423f3640b7beb79af13e417687bf67746cbf35cd71bfb852b0be95ee07bfc29e5cbe7c6d8cbf303db23fc10e3f3f47cf4c3e0e64e83e581f24bfbb24993f9893c73de12adcbe759a5fbe471b89be99cddbbf515cb6be6ecb8a3fe182cb3e8cfe06bf4231a63b6149fd3d8223f8bf3cf8ed3c81e8dfbd76c5983f5298d4be7c01fd3e3c33993f4bd5943fd460d53f68206c40805f0c40ee94543e13831dbfcfc5733f63ca3dbf1761a4be131e93bf0d38b7bf8cda8a3f0ae9033feefae23f4e1d67bf613dcfbe79b7d7bfdbd15bbd7d48493f85a9c03d37fd79bf7b06f33ed9f50bbebb648e3d7d241e3f07448dbf1dc7443e882023be24acfdbf24b3a43ea1c78abe46effd3e6b705bbffe8f893ffd0512bec7075f3fd420ee3ef54eb4bfcb55b33ce883debf380acfbfa425f93ea8d6093f05caf4bee2c37dbf2d6fccbf863532be00873bbf9aab23beebbc28bfe859213f815f823ff9af80bfd68c32bf219da13fd9dda7bf076af73ca82428beb981a9bef3debd3cb133103fed204d3f1942f53f93798cbe889430bfcfd4c03f9b35563e3c6ba73fac61adbe1f54963f0a58f3bf4176403fadce0d3fb074a23fa3c6e03fdd7b5f3fbc4d813f5568b73e87d297bffa4c8a3fe65e99bf14aee8bffde5873d7848e0bedcb89d3f643e0c3f8b81383fcae734c09bf1573e0f85154077b8673e3226473fc76fc2bf870c9c3eee26ca3e745c2dbeb8f6f2bf9a11533fea768bbfe7a3ec3f93791a3fca63c93addbf713fea7bc5bde8650140290506bf4e9ab53f66e55dbdfceab73e643d0dbfe2ed98bfdde934bf4112a2bf5a091c3f23aa86bf25d007bfdbfc76bf379e28bf532ba03e7ab5fbbff33652bf597e0c3ff9930240b5058f3c9dee2ebf4ac5ddbe102522bd2ce93c3dd09a1bc01c5f54403e084fbfb524103f361b963fb2b26fbe7c7309409083813f12d18bbeca25cf3fb199be3fa560763f0a82a23d"), dtype=np.float32).reshape([1, 3, 32, 32])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
