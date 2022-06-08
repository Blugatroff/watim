(module
	(import "wasi_unstable" "proc_exit" (func $exit (param $code i32)))
	(import "wasi_unstable" "fd_read" (func $raw_read (param $file i32) (param $iovs i32) (param $iovs_count i32) (param $result i32) (result i32)))
	(import "wasi_unstable" "fd_write" (func $raw_write (param $file i32) (param $ptr i32) (param $len i32) (param $nwritten i32) (result i32)))

	(memory 1)
	(export "memory" (memory 0))
	(global $stac:k (mut i32) (i32.const 21))
	(data (i32.const 0) "Failed to parse: ''\n")
	
	(func $read (param $file i32) (param $buf_addr i32) (param $buf_size i32) (result i32)	
		(local $space i32)	
		(local $stac:k i32)
		global.get $stac:k
		local.set $stac:k
		global.get $stac:k
		i32.const 4
		global.get $stac:k
		i32.const 4
		i32.rem_u
		i32.sub
		i32.add
		global.set $stac:k
		global.get $stac:k
		global.get $stac:k
		i32.const 8
		i32.add
		global.set $stac:k
		local.set $space	
		local.get $space
		local.get $buf_addr
		i32.store
		local.get $space
		i32.const 4
		i32.add
		local.get $buf_size
		i32.store
		local.get $file
		local.get $space
		i32.const 1
		local.get $space
		call $raw_read
		drop
		local.get $space
		i32.load
		local.get $stac:k
		global.set $stac:k
	)
	(func $write (param $file i32) (param $ptr i32) (param $len i32) (result i32)	
		(local $space i32)	
		(local $stac:k i32)
		global.get $stac:k
		local.set $stac:k
		global.get $stac:k
		i32.const 4
		global.get $stac:k
		i32.const 4
		i32.rem_u
		i32.sub
		i32.add
		global.set $stac:k
		global.get $stac:k
		global.get $stac:k
		i32.const 8
		i32.add
		global.set $stac:k
		local.set $space	
		local.get $space
		local.get $ptr
		i32.store
		local.get $space
		i32.const 4
		i32.add
		local.get $len
		i32.store
		local.get $file
		local.get $space
		i32.const 1
		local.get $space
		call $raw_write
		local.get $stac:k
		global.set $stac:k
	)
	(func $print (param $n i32)	
		(local $l i32)
		(local $i i32)
		(local $buf i32)
		(local $buf_reversed i32)	
		(local $stac:k i32)
		global.get $stac:k
		local.set $stac:k
		global.get $stac:k
		global.get $stac:k
		i32.const 16
		i32.add
		global.set $stac:k
		local.set $buf
		global.get $stac:k
		global.get $stac:k
		i32.const 16
		i32.add
		global.set $stac:k
		local.set $buf_reversed	
		i32.const 1
		i32.const 100
		i32.const 7
		call $write
		drop
		i32.const 0
		local.set $l
		local.get $n
		i32.const 0
		i32.eq
		(if 
			(then
				i32.const 1
				local.set $l
				local.get $buf
				i32.const 48
				i32.store8
			)	
			(else
				(block $block 
					(loop $loop 
						local.get $n
						i32.const 0
						i32.eq
						(if 
							(then
								br $block
							)
						)
						local.get $buf
						local.get $l
						i32.add
						local.get $n
						i32.const 10
						i32.rem_u
						i32.const 48
						i32.add
						i32.store8
						local.get $n
						i32.const 10
						i32.div_u
						local.set $n
						local.get $l
						i32.const 1
						i32.add
						local.set $l
						br $loop
					)
				)
			)
		)
		local.get $l
		local.set $i
		(block $block 
			(loop $loop 
				local.get $buf_reversed
				local.get $l
				i32.const 1
				i32.sub
				local.get $i
				i32.sub
				i32.add
				local.get $buf
				local.get $i
				i32.add
				i32.load
				i32.store
				local.get $i
				i32.const 0
				i32.eq
				(if 
					(then
						br $block
					)
				)
				local.get $i
				i32.const 1
				i32.sub
				local.set $i
				br $loop
			)
		)
		local.get $buf_reversed
		local.get $l
		i32.add
		i32.const 10
		i32.store8
		i32.const 1
		local.get $buf_reversed
		local.get $l
		i32.const 1
		i32.add
		call $write
		drop
		local.get $stac:k
		global.set $stac:k
	)
	(func $parse (param $ptr i32) (param $len i32) (result i32)	
		(local $n i32)
		(local $d i32)
		(local $original-ptr i32)
		(local $original-len i32)	
		local.get $ptr
		local.set $original-ptr
		local.get $len
		local.set $original-len
		(block $block (result i32)
			(loop $loop (result i32)
				local.get $ptr
				i32.load8_u
				local.set $d
				local.get $d
				i32.const 48
				i32.ge_u
				local.get $d
				i32.const 58
				i32.le_u
				i32.and
				(if 
					(then
						local.get $n
						local.get $d
						i32.const 48
						i32.sub
						i32.add
						local.set $n
					)	
					(else
						i32.const 1
						i32.const 0
						i32.const 18
						call $write
						drop
						i32.const 1
						local.get $original-ptr
						local.get $original-len
						call $write
						drop
						i32.const 1
						i32.const 18
						i32.const 3
						call $write
						drop
						i32.const 1
						call $exit
					)
				)
				local.get $ptr
				i32.const 1
				i32.add
				local.set $ptr
				local.get $len
				i32.const 1
				i32.sub
				local.set $len
				local.get $len
				i32.const 0
				i32.eq
				(if 
					(then
						local.get $n
						br $block
					)
				)
				local.get $n
				i32.const 10
				i32.mul
				local.set $n
				br $loop
			)
		)
	)
	(func $dup (param $a i32) (result i32 i32)	
		local.get $a
		local.get $a
	)
	(func $main (export "_start") 	
		(local $nread i32)
		(local $buf i32)	
		(local $stac:k i32)
		global.get $stac:k
		local.set $stac:k
		global.get $stac:k
		global.get $stac:k
		i32.const 32
		i32.add
		global.set $stac:k
		local.set $buf	
		i32.const 0
		local.get $buf
		i32.const 32
		call $read
		local.set $nread
		local.get $buf
		local.get $nread
		i32.const 1
		i32.sub
		call $parse
		call $dup
		call $print
		call $exit
		local.get $stac:k
		global.set $stac:k
	)
)
